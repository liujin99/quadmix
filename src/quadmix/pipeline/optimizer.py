"""
QuaDMix parameter optimizer — LightGBM regression + optimal parameter search.

This module implements:
  1. LightGBM regressor training from proxy experiment results
  2. Optimal parameter search over 100,000 sampled configurations
  3. Top-K averaging for variance reduction

Based on Sections 3.2 and 3.3 of the paper.
"""

from typing import List, Optional, Dict, Any, Tuple
import warnings
import numpy as np
import numpy.typing as npt
from quadmix.core.types import ParameterSet, QuaDMixConfig, ProxyResult
from quadmix.pipeline.param_sampler import ParameterSampler


def _bootstrap_one(
    seed: int,
    params_list: List[ParameterSet],
    losses: npt.NDArray[np.float64],
    n_features: int,
    num_domains: int,
    num_criteria: int,
    regression_params: dict,
) -> Optional[Tuple[float, "RegressionModel"]]:
    n_total = len(params_list)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_total, replace=True)
    unique_idx = np.unique(idx)

    oob_mask = np.ones(n_total, dtype=bool)
    oob_mask[unique_idx] = False
    oob_idx = np.where(oob_mask)[0]

    if len(unique_idx) < max(10, n_features // 2) or len(oob_idx) < 5:
        return None

    boot_train_params = [params_list[j] for j in idx]
    boot_train_losses = losses[idx]
    oob_params = [params_list[j] for j in oob_idx]
    oob_losses = losses[oob_idx]

    try:
        model = RegressionModel(
            model_type="lightgbm",
            n_jobs=1,
            **regression_params,
        )
        model.fit(
            boot_train_params,
            boot_train_losses,
            num_domains=num_domains,
            num_criteria=num_criteria,
            eval_params_list=oob_params,
            eval_losses=oob_losses,
        )
        r2 = float(model.score(oob_params, oob_losses))
        return (r2, model)
    except Exception:
        return None


class RegressionModel:
    """
    Wrapper around LightGBM regressor for QuaDMix parameter search.

    The regressor R maps: R(θ_i) → L_i (predicted validation loss)
    where θ_i is the flattened parameter vector and L_i is the
    corresponding proxy model validation loss.
    """

    def __init__(self, model_type: str = "lightgbm", **model_kwargs):
        """
        Args:
            model_type: Regression model type.
                        Options: 'lightgbm', 'random_forest', 'linear'.
            **model_kwargs: Additional kwargs passed to the regressor constructor.
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self._model = None
        self._is_fitted = False

    def _build_model(self, n_train: int = 0):
        """Build the underlying regressor model."""
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError(
                    "LightGBM is required. Install with: pip install lightgbm"
                )

            if n_train > 0 and n_train < 500:
                max_depth = min(5, max(3, int(np.log2(n_train))))
                default_params = {
                    "n_estimators": 500,
                    "learning_rate": 0.02,
                    "max_depth": max_depth,
                    "num_leaves": min(15, 2 ** max_depth - 1),
                    "min_child_samples": max(30, n_train // 6),
                    "subsample": 0.8,
                    "colsample_bytree": 0.6,
                    "reg_alpha": 1.0,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "verbose": -1,
                }
            else:
                default_params = {
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_child_samples": min(20, max(5, n_train // 10)),
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "random_state": 42,
                    "verbose": -1,
                }
            default_params.update(self.model_kwargs)
            self._model = lgb.LGBMRegressor(**default_params)

        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            default_params = {
                "n_estimators": 500,
                "max_depth": 20,
                "min_samples_leaf": 5,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_kwargs)
            self._model = RandomForestRegressor(**default_params)

        elif self.model_type == "linear":
            from sklearn.linear_model import Ridge

            default_params = {
                "alpha": 1.0,
                "random_state": 42,
            }
            default_params.update(self.model_kwargs)
            self._model = Ridge(**default_params)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        params_list: List[ParameterSet],
        losses: npt.NDArray[np.float64],
        num_domains: int,
        num_criteria: int,
        eval_params_list: Optional[List[ParameterSet]] = None,
        eval_losses: Optional[npt.NDArray[np.float64]] = None,
        early_stopping_rounds: int = 50,
    ) -> "RegressionModel":
        """
        Train the regression model on proxy experiment results.

        Args:
            params_list: List of parameter configurations used in proxy experiments.
            losses: Corresponding validation losses. Shape: (n_experiments,).
            num_domains: Number of domains M.
            num_criteria: Number of quality criteria N.
            eval_params_list: Optional validation parameter configurations.
            eval_losses: Optional validation losses.
            early_stopping_rounds: Early stopping patience (LightGBM only).

        Returns:
            Self (fitted model).
        """
        if len(params_list) == 0:
            raise ValueError("Empty params_list provided for regression fitting")

        # Flatten parameters into feature vectors
        X = np.array([p.flatten() for p in params_list])
        y = np.array(losses)

        self._num_domains = num_domains
        self._num_criteria = num_criteria

        self._build_model(n_train=len(params_list))

        if (self.model_type == "lightgbm" and eval_params_list is not None
                and eval_losses is not None and len(eval_params_list) > 0):
            X_val = np.array([p.flatten() for p in eval_params_list])
            y_val = np.array(eval_losses)
            import lightgbm as lgb
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._model.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, params_list: List[ParameterSet]) -> npt.NDArray[np.float64]:
        """
        Predict validation losses for given parameter configurations.

        Args:
            params_list: List of parameter configurations to evaluate.

        Returns:
            Array of predicted losses. Shape: (len(params_list),).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X = np.array([p.flatten() for p in params_list])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self._model.predict(X)

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the fitted model (LightGBM/RF only)."""
        if not self._is_fitted or self._model is None:
            return None
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            names = []
            N, M = self._num_criteria, self._num_domains
            # Domain weights (no global weights — they are intermediate in Algorithm 1)
            for m in range(M):
                for n in range(N):
                    names.append(f"alpha_{m}_{n}")
            # Sampling params
            for m in range(M):
                names.append(f"lambda_{m}")
                names.append(f"omega_{m}")
                names.append(f"eta_{m}")
                names.append(f"epsilon_{m}")

            return dict(zip(names[: len(importances)], importances))
        return None

    def score(self, params_list: List[ParameterSet], losses: npt.NDArray[np.float64]) -> float:
        """Return R² score for the model on given data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet.")
        X = np.array([p.flatten() for p in params_list])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return float(self._model.score(X, losses))

    def save(self, path: str):
        """Save the regression model to disk."""
        import joblib
        data = {
            "model": self._model,
            "model_type": self.model_type,
            "model_kwargs": self.model_kwargs,
            "num_domains": self._num_domains,
            "num_criteria": self._num_criteria,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "RegressionModel":
        """Load a fitted regression model from disk."""
        import joblib
        data = joblib.load(path)
        model = cls(model_type=data["model_type"], **data["model_kwargs"])
        model._model = data["model"]
        model._num_domains = data["num_domains"]
        model._num_criteria = data["num_criteria"]
        model._is_fitted = True
        return model


class QuaDMixOptimizer:
    """
    Full QuaDMix parameter optimizer.

    Pipeline:
      1. Collect proxy experiment results (params → loss)
      2. Train regression model (LightGBM)
      3. Search for optimal parameters
      4. Return top-K averaged parameters
    """

    def __init__(
        self,
        config: QuaDMixConfig,
        regression_params: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.regression_params = regression_params or {}
        self._regressor: Optional[RegressionModel] = None
        self._proxy_results: List[ProxyResult] = []
        self._per_task_models: Optional[Dict[str, RegressionModel]] = None
        self._per_task_weights: Optional[Dict[str, float]] = None
        self._per_task_r2: Optional[Dict[str, float]] = None
        self._per_task_train_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self._aggregate_train_stats: Optional[Tuple[float, float]] = None
        self._ensemble_val_r2: Optional[float] = None
        self._ensemble_val_mae: Optional[float] = None

    def add_proxy_results(self, results: List[ProxyResult]):
        """Add proxy experiment results."""
        self._proxy_results.extend(results)

    def train_regressor(self) -> RegressionModel:
        """
        Train the LightGBM regressor on collected proxy results.

        Data split: ~2800/200 train/validation (matching paper's 3000 total).
        """
        if len(self._proxy_results) < 2:
            raise ValueError(
                f"Need at least 2 proxy results, got {len(self._proxy_results)}"
            )

        # Prepare data
        params_list = [r.parameters for r in self._proxy_results]
        losses = np.array([r.validation_loss for r in self._proxy_results], dtype=np.float64)

        # Filter out inf/nan values
        valid_mask = np.isfinite(losses)
        if not np.all(valid_mask):
            invalid_count = (~valid_mask).sum()
            print(f"[QuaDMixOptimizer] WARNING: filtering {invalid_count} experiments with non-finite val_loss")
            params_list = [p for p, v in zip(params_list, valid_mask) if v]
            losses = losses[valid_mask]

        # Split into train/validation (skip split if too few samples)
        n_total = len(params_list)
        if n_total < 10:
            train_params, train_losses = params_list, losses
            val_params, val_losses = [], np.array([], dtype=np.float64)
            train_idx = np.arange(n_total)
            val_idx = np.array([], dtype=np.int64)
            n_val = 0
            print(f"[QuaDMixOptimizer] Only {n_total} experiments, using all for training")
        else:
            # Dynamic adjustment: ensure validation set has at least 20 samples (or 20%, whichever is smaller)
            # This prevents too-small validation sets when n_total is small (e.g., 64 experiments)
            min_val = min(20, n_total // 5)
            n_val = max(min_val, int(n_total * (1 - self.config.regression_train_ratio)))
            n_train = n_total - n_val
            
            rng = np.random.default_rng(42)
            indices = rng.permutation(n_total)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            train_params = [params_list[i] for i in train_idx]
            train_losses = losses[train_idx]
            val_params = [params_list[i] for i in val_idx]
            val_losses = losses[val_idx]
            n_val = len(val_params)

        # Train regressor
        self._regressor = RegressionModel(
            model_type="lightgbm",
            **self.regression_params,
        )
        self._regressor.fit(
            train_params,
            train_losses,
            num_domains=self.config.num_domains,
            num_criteria=self.config.num_quality_criteria,
            eval_params_list=val_params if n_val > 0 else None,
            eval_losses=val_losses if n_val > 0 else None,
        )

        # Evaluate (handle tiny experiment counts)
        train_r2 = float(self._regressor.score(train_params, train_losses))
        if n_val > 0:
            val_pred = self._regressor.predict(val_params)
            val_mae = float(np.mean(np.abs(val_pred - val_losses)))
            val_r2 = float(self._regressor.score(val_params, val_losses))
        else:
            val_pred = np.array([])
            val_mae = 0.0
            val_r2 = 0.0

        self._train_r2 = train_r2
        self._val_mae = val_mae
        self._val_r2 = val_r2

        print(f"[QuaDMixOptimizer] Aggregate model (diagnostic — not used for search):")
        print(f"  Train R² = {train_r2:.4f}")
        if n_val > 0:
            print(f"  Val   R² = {val_r2:.4f}, MAE = {val_mae:.4f}")
        print(f"  Val   samples = {n_val}")

        self._compute_reliability(params_list, losses)

        # Train per-task models if per-task losses are available
        has_per_task = all(r.per_task_losses is not None for r in self._proxy_results)
        if has_per_task:
            self._train_per_task_models(
                params_list, losses, train_idx, val_idx
            )

        return self._regressor

    def _train_per_task_models(
        self,
        params_list: List[ParameterSet],
        losses: npt.NDArray[np.float64],
        train_idx: npt.NDArray[np.int64],
        val_idx: npt.NDArray[np.int64],
    ):
        """Train per-task LightGBM models with R²-adaptive weights and z-score calibration."""
        tasks = sorted(self._proxy_results[0].per_task_losses.keys())
        
        all_task_losses = {}
        for task in tasks:
            task_losses = np.array([r.per_task_losses[task] for r in self._proxy_results], dtype=np.float64)
            all_task_losses[task] = task_losses
        
        task_variances = {task: np.var(tl) for task, tl in all_task_losses.items()}
        valid_tasks = {task: var for task, var in task_variances.items() if var > 1e-8}
        
        if len(valid_tasks) < len(task_variances):
            excluded = set(task_variances.keys()) - set(valid_tasks.keys())
            print(f"[QuaDMixOptimizer] WARNING: Excluding {len(excluded)} zero-variance tasks: {excluded}")
        
        if len(valid_tasks) == 0:
            print("[QuaDMixOptimizer] ERROR: All tasks have zero variance, skipping per-task models")
            return
        
        self._per_task_models = {}
        self._per_task_r2 = {}
        self._per_task_train_r2 = {}
        self._per_task_train_stats = {}
        
        train_losses = losses[train_idx]
        self._aggregate_train_stats = (float(np.mean(train_losses)), float(np.std(train_losses)))
        
        print(f"[QuaDMixOptimizer] Training {len(valid_tasks)} per-task models...")
        
        for task in valid_tasks:
            task_losses = all_task_losses[task]
            task_train_losses = task_losses[train_idx]
            task_val_losses = task_losses[val_idx] if len(val_idx) > 0 else None
            
            self._per_task_train_stats[task] = (
                float(np.mean(task_train_losses)),
                float(np.std(task_train_losses)),
            )
            
            train_params = [params_list[i] for i in train_idx]
            val_params = [params_list[i] for i in val_idx] if len(val_idx) > 0 else None
            
            model = RegressionModel(model_type="lightgbm", **self.regression_params)
            model.fit(
                train_params,
                task_train_losses,
                num_domains=self.config.num_domains,
                num_criteria=self.config.num_quality_criteria,
                eval_params_list=val_params,
                eval_losses=task_val_losses,
            )
            
            self._per_task_models[task] = model
            
            task_train_r2 = float(model.score(train_params, task_train_losses))
            self._per_task_train_r2[task] = task_train_r2
            
            if len(val_idx) > 0:
                task_r2 = float(model.score(val_params, task_val_losses))
            else:
                task_r2 = task_train_r2
            self._per_task_r2[task] = task_r2
        
        task_stds = {task: np.sqrt(var) for task, var in valid_tasks.items()}
        raw_weights = {}
        for task in valid_tasks:
            r2 = self._per_task_r2[task]
            raw_weights[task] = max(r2, 0.0) * task_stds[task]
        
        total_raw = sum(raw_weights.values())
        if total_raw < 1e-12:
            print("[QuaDMixOptimizer] WARNING: All tasks have R²<=0, falling back to std-only weights")
            total_std = sum(task_stds.values())
            self._per_task_weights = {task: std / total_std for task, std in task_stds.items()}
        else:
            self._per_task_weights = {task: w / total_raw for task, w in raw_weights.items()}
        
        n_filtered = sum(1 for task in valid_tasks if self._per_task_r2[task] <= 0)
        active_tasks = [t for t in valid_tasks if self._per_task_weights[t] > 0]
        
        print(f"[QuaDMixOptimizer] Per-task models trained ({len(active_tasks)} active, {n_filtered} filtered):")
        print(f"  {'Task':<30} {'Train R²':>10} {'Val R²':>10} {'Gap':>8} {'Weight':>8} {'Std':>8}")
        print(f"  {'-'*76}")
        for task in sorted(valid_tasks.keys(), key=lambda t: -self._per_task_weights[t]):
            train_r2 = self._per_task_train_r2[task]
            val_r2 = self._per_task_r2[task]
            gap = train_r2 - val_r2
            weight = self._per_task_weights[task]
            std = task_stds[task]
            marker = " [filtered]" if weight == 0 else ""
            print(f"  {task:<30} {train_r2:>10.4f} {val_r2:>10.4f} {gap:>8.3f} {weight:>8.4f} {std:>8.4f}{marker}")

        if len(val_idx) > 0:
            agg_mean, agg_std = self._aggregate_train_stats
            val_losses_actual = losses[val_idx]
            val_params = [params_list[i] for i in val_idx]
            z_score_sum = np.zeros(len(val_idx))
            for task, model in self._per_task_models.items():
                w = self._per_task_weights[task]
                if w <= 0:
                    continue
                task_mean, task_std = self._per_task_train_stats[task]
                raw_pred = model.predict(val_params)
                z_pred = (raw_pred - task_mean) / max(task_std, 1e-8)
                z_score_sum += w * z_pred
            ensemble_pred = z_score_sum * agg_std + agg_mean
            ensemble_r2 = float(1 - np.sum((ensemble_pred - val_losses_actual) ** 2) / max(np.sum((val_losses_actual - np.mean(val_losses_actual)) ** 2), 1e-12))
            ensemble_mae = float(np.mean(np.abs(ensemble_pred - val_losses_actual)))
            self._ensemble_val_r2 = ensemble_r2
            self._ensemble_val_mae = ensemble_mae
            print(f"[QuaDMixOptimizer] Ensemble Val R² = {ensemble_r2:.4f}, MAE = {ensemble_mae:.4f} ({len(active_tasks)} active tasks)")
        else:
            self._ensemble_val_r2 = None
            self._ensemble_val_mae = None

    def _compute_reliability(
        self,
        params_list: List[ParameterSet],
        losses: npt.NDArray[np.float64],
    ):
        n_total = len(params_list)
        n_features = len(params_list[0].flatten()) if n_total > 0 else 0

        self._n_features = n_features
        self._n_train = n_total
        self._n_val = None
        self._sample_sufficient = n_total >= n_features
        self._overfit_gap = self._train_r2 - self._val_r2 if self._val_r2 else None

        self._val_r2_ci_lower = None
        self._val_r2_ci_upper = None
        self._val_r2_ci_std = None
        self._val_r2_bootstrap_mean = None
        self._bootstrap_models: List[RegressionModel] = []

        if n_total < 10:
            print(f"[QuaDMixOptimizer] ⚠️  Too few samples ({n_total} < 10), skip bootstrap")
            return

        if n_total < 50:
            n_bootstrap = 100
        elif n_total < 200:
            n_bootstrap = 200
        elif n_total < 500:
            n_bootstrap = 300
        else:
            n_bootstrap = 500
        n_ensemble = min(50, n_bootstrap // 4)

        print(f"[QuaDMixOptimizer] Bootstrap: {n_bootstrap} iterations (parallel, full resampling, OOB evaluation)...")
        from joblib import Parallel, delayed
        import os
        n_jobs = min(os.cpu_count() or 4, n_bootstrap)
        seeds = np.random.default_rng(42).integers(0, 2**31, size=n_bootstrap).tolist()

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_bootstrap_one)(
                seed=int(seeds[i]),
                params_list=params_list,
                losses=losses,
                n_features=n_features,
                num_domains=self.config.num_domains,
                num_criteria=self.config.num_quality_criteria,
                regression_params=self.regression_params,
            )
            for i in range(n_bootstrap)
        )

        bootstrap_r2s = []
        for res in results:
            if res is not None:
                r2, model = res
                bootstrap_r2s.append(r2)
                if len(self._bootstrap_models) < n_ensemble:
                    self._bootstrap_models.append(model)

        if len(bootstrap_r2s) < 10:
            print(f"[QuaDMixOptimizer] ⚠️  Bootstrap failed ({len(bootstrap_r2s)} valid < 10), skip all")
            return

        bootstrap_r2s = np.array(bootstrap_r2s)
        self._val_r2_bootstrap_mean = float(np.mean(bootstrap_r2s))
        print(f"[QuaDMixOptimizer] Val   R² (bootstrap mean) = {self._val_r2_bootstrap_mean:.4f} ({len(bootstrap_r2s)} iterations)")
        print(f"[QuaDMixOptimizer] Bootstrap ensemble: {len(self._bootstrap_models)} models trained")

        if len(bootstrap_r2s) < 50:
            print(f"[QuaDMixOptimizer] ⚠️  Too few valid iterations ({len(bootstrap_r2s)} < 50), skip CI")
            return

        self._val_r2_ci_lower = float(np.percentile(bootstrap_r2s, 2.5))
        self._val_r2_ci_upper = float(np.percentile(bootstrap_r2s, 97.5))
        self._val_r2_ci_std = float(np.std(bootstrap_r2s))

        ci_width = self._val_r2_ci_upper - self._val_r2_ci_lower
        status = "✓ Stable" if ci_width < 0.3 else "⚠️ Wide CI"

        print(f"[QuaDMixOptimizer] Val   R² 95% CI = [{self._val_r2_ci_lower:.3f}, {self._val_r2_ci_upper:.3f}] (width={ci_width:.3f}) {status}")

        if not self._sample_sufficient:
            print(f"[QuaDMixOptimizer] ⚠️  Samples ({n_total}) < Features ({n_features}): underdetermined, increase experiments to {n_features * 3}+")
        elif self._overfit_gap and self._overfit_gap > 0.3:
            print(f"[QuaDMixOptimizer] ⚠️  Train-Val gap = {self._overfit_gap:.3f}: possible overfitting")

    def search_optimal(
        self,
        n_search_points: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> tuple[ParameterSet, List[ParameterSet], npt.NDArray[np.float64]]:
        """
        Search for optimal parameters using the trained regressor.

        Based on Section 3.3: sample 100,000 points, predict loss,
        average top-K.

        If bootstrap ensemble models are available, uses ensemble prediction
        (average of all bootstrap models) instead of single model for more
        stable results.

        Args:
            n_search_points: Number of candidate points to sample.
                             Default: config.num_search_points.
            top_k: Number of top candidates to average.
                   Default: config.top_k_average.

        Returns:
            Tuple of (optimal_params, all_candidates, candidate_losses).
        """
        if self._regressor is None:
            raise RuntimeError("Regressor not trained. Call train_regressor() first.")

        n_search = n_search_points or self.config.num_search_points
        k = top_k or self.config.top_k_average

        # Sample search points using the same parameter sampler
        sampler = ParameterSampler(self.config, seed=9999)
        candidates = sampler.sample_batch(n_search)

        # Predict losses using per-task weighted prediction if available
        if self._per_task_models and self._per_task_weights and self._per_task_train_stats and self._aggregate_train_stats:
            agg_mean, agg_std = self._aggregate_train_stats
            z_score_sum = np.zeros(n_search)
            active_count = 0
            
            for task, model in self._per_task_models.items():
                w = self._per_task_weights[task]
                if w <= 0:
                    continue
                task_mean, task_std = self._per_task_train_stats[task]
                raw_pred = model.predict(candidates)
                z_pred = (raw_pred - task_mean) / max(task_std, 1e-8)
                z_score_sum += w * z_pred
                active_count += 1
            
            predicted_losses = z_score_sum * agg_std + agg_mean
            print(f"[QuaDMixOptimizer] Search: per-task z-score calibrated prediction ({active_count} active tasks)")
        elif hasattr(self, '_bootstrap_models') and len(self._bootstrap_models) > 0:
            all_preds = np.array([m.predict(candidates) for m in self._bootstrap_models])
            predicted_losses = np.mean(all_preds, axis=0)
            print(f"[QuaDMixOptimizer] Search: ensemble prediction ({len(self._bootstrap_models)} models)")
        else:
            predicted_losses = self._regressor.predict(candidates)
            print(f"[QuaDMixOptimizer] Search: single model prediction")

        # Find top-K
        top_indices = np.argsort(predicted_losses)[:k]

        # Average top-K to get final parameters
        N, M = self.config.num_quality_criteria, self.config.num_domains
        avg_arr = np.mean([candidates[i].flatten() for i in top_indices], axis=0)
        optimal_params = ParameterSet.from_flattened(avg_arr, M, N)

        return optimal_params, candidates, predicted_losses

    @property
    def regressor(self) -> Optional[RegressionModel]:
        return self._regressor

    @property
    def train_r2(self) -> Optional[float]:
        return getattr(self, "_train_r2", None)

    @property
    def val_r2(self) -> Optional[float]:
        return getattr(self, "_val_r2", None)

    @property
    def val_mae(self) -> Optional[float]:
        return getattr(self, "_val_mae", None)

    @property
    def val_r2_ci_lower(self) -> Optional[float]:
        return getattr(self, "_val_r2_ci_lower", None)

    @property
    def val_r2_ci_upper(self) -> Optional[float]:
        return getattr(self, "_val_r2_ci_upper", None)

    @property
    def val_r2_bootstrap_mean(self) -> Optional[float]:
        return getattr(self, "_val_r2_bootstrap_mean", None)

    @property
    def ensemble_val_r2(self) -> Optional[float]:
        return self._ensemble_val_r2

    @property
    def ensemble_val_mae(self) -> Optional[float]:
        return self._ensemble_val_mae

    @property
    def sample_sufficient(self) -> Optional[bool]:
        return getattr(self, "_sample_sufficient", None)

    @property
    def overfit_gap(self) -> Optional[float]:
        return getattr(self, "_overfit_gap", None)

    @property
    def n_features(self) -> Optional[int]:
        return getattr(self, "_n_features", None)

    @property
    def per_task_analysis(self) -> Optional[Dict[str, Any]]:
        if not self._per_task_models or not self._per_task_weights or not self._per_task_r2:
            return None
        tasks = []
        for task in sorted(self._per_task_weights.keys(), key=lambda t: -self._per_task_weights[t]):
            tasks.append({
                "name": task,
                "r2": self._per_task_r2[task],
                "weight": self._per_task_weights[task],
                "std": self._per_task_train_stats[task][1] if self._per_task_train_stats else None,
            })
        return {
            "tasks": tasks,
            "n_active": sum(1 for t in tasks if t["weight"] > 0),
            "n_filtered": sum(1 for t in tasks if t["weight"] == 0),
        }