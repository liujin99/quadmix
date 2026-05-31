"""
QuaDMix parameter optimizer — LightGBM regression + optimal parameter search.

This module implements:
  1. LightGBM regressor training from proxy experiment results
  2. Optimal parameter search over 100,000 sampled configurations
  3. Top-K averaging for variance reduction

Based on Sections 3.2 and 3.3 of the paper.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import numpy.typing as npt
from quadmix.core.types import ParameterSet, QuaDMixConfig, ProxyResult
from quadmix.pipeline.param_sampler import ParameterSampler


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

    def _build_model(self):
        """Build the underlying regressor model."""
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError(
                    "LightGBM is required. Install with: pip install lightgbm"
                )

            default_params = {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
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
    ) -> "RegressionModel":
        """
        Train the regression model on proxy experiment results.

        Args:
            params_list: List of parameter configurations used in proxy experiments.
            losses: Corresponding validation losses. Shape: (n_experiments,).
            num_domains: Number of domains M.
            num_criteria: Number of quality criteria N.

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

        self._build_model()
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
        return self._model.predict(X)

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the fitted model (LightGBM/RF only)."""
        if not self._is_fitted or self._model is None:
            return None
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            names = []
            N, M = self._num_criteria, self._num_domains
            # Global weights
            for n in range(N):
                names.append(f"global_weight_criterion_{n}")
            # Domain weights
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
            n_val = 0
            print(f"[QuaDMixOptimizer] Only {n_total} experiments, using all for training")
        else:
            n_train = int(n_total * self.config.regression_train_ratio)
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

        print(f"[QuaDMixOptimizer] Train R² = {train_r2:.4f}")
        if n_val > 0:
            print(f"[QuaDMixOptimizer] Val   R² = {val_r2:.4f}, MAE = {val_mae:.4f}")
        print(f"[QuaDMixOptimizer] Val   samples = {n_val}")

        return self._regressor

    def search_optimal(
        self,
        n_search_points: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> tuple[ParameterSet, List[ParameterSet], npt.NDArray[np.float64]]:
        """
        Search for optimal parameters using the trained regressor.

        Based on Section 3.3: sample 100,000 points, predict loss,
        average top-K.

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

        # Predict losses
        predicted_losses = self._regressor.predict(candidates)

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