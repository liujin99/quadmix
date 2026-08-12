"""RegressionModel — LightGBM/RF/Ridge wrapper for QuaDMix parameter search.

Extracted from optimizer.py to isolate the model concern from the search logic.
"""

from typing import Dict, List, Optional
import warnings
import numpy as np
import numpy.typing as npt
from quadmix.core.types import ParameterSet


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

    def _build_model(self, n_train: int = 0, n_features: int = 0, verbose: bool = True):
        """Build the underlying regressor model."""
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError(
                    "LightGBM is required. Install with: pip install lightgbm"
                )

            ratio = n_train / max(1, n_features) if n_features > 0 else float("inf")
            if verbose:
                regime = "conservative" if ratio < 3 else ("moderate" if ratio < 8 else "aggressive")
                print(f"[LightGBM] n/p ratio={ratio:.1f} → {regime} regime (n={n_train}, p={n_features})")

            if ratio < 3:
                max_depth = min(4, max(2, int(np.log2(max(2, n_train)))))
                default_params = {
                    "n_estimators": 300,
                    "learning_rate": 0.01,
                    "max_depth": max_depth,
                    "num_leaves": min(7, 2 ** max_depth - 1),
                    "min_child_samples": max(5, int(np.ceil(np.sqrt(n_train))), n_features // 5),
                    "subsample": 0.7,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 5.0,
                    "reg_lambda": 5.0,
                    "random_state": 42,
                    "verbose": -1,
                }
            elif ratio < 8:
                max_depth = min(5, max(3, int(np.log2(max(2, n_train)))))
                default_params = {
                    "n_estimators": 500,
                    "learning_rate": 0.02,
                    "max_depth": max_depth,
                    "num_leaves": min(15, 2 ** max_depth - 1),
                    "min_child_samples": max(5, int(np.ceil(np.sqrt(n_train))), n_features // 5),
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
                    "min_child_samples": max(5, int(np.ceil(np.sqrt(n_train))), n_features // 5),
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
        verbose: bool = True,
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
            verbose: Whether to print regime info.

        Returns:
            Self (fitted model).
        """
        if len(params_list) == 0:
            raise ValueError("Empty params_list provided for regression fitting")

        X = np.array([p.flatten() for p in params_list])
        y = np.array(losses)

        self._num_domains = num_domains
        self._num_criteria = num_criteria

        n_features = X.shape[1]
        self._build_model(n_train=len(params_list), n_features=n_features, verbose=verbose)

        if (self.model_type == "lightgbm" and eval_params_list is not None
                and eval_losses is not None and len(eval_params_list) > 0):
            X_val = np.array([p.flatten() for p in eval_params_list])
            y_val = np.array(eval_losses)
            import lightgbm as lgb
            import inspect
            use_eval_xy = 'eval_X' in inspect.signature(
                lgb.LGBMRegressor.fit).parameters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_kwargs = dict(
                    callbacks=[lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds, verbose=False)],
                )
                if use_eval_xy:
                    fit_kwargs['eval_X'] = X_val
                    fit_kwargs['eval_y'] = y_val
                else:
                    fit_kwargs['eval_set'] = [(X_val, y_val)]
                self._model.fit(X, y, **fit_kwargs)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
            for m in range(M):
                for n in range(N):
                    names.append(f"alpha_{m}_{n}")
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
