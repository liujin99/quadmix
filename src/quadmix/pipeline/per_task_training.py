"""Per-task training and bootstrap workers for QuaDMix optimizer.

Standalone functions (multiprocessing-safe, no `self`) extracted from
optimizer.py. Called by QuaDMixOptimizer._train_per_task_models and
_compute_reliability.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from quadmix.core.types import ParameterSet
from quadmix.pipeline.regression_model import RegressionModel


def train_single_task_cv_fold(
    fold_idx: int,
    folds: List[npt.NDArray[np.int64]],
    n_folds: int,
    task_losses: npt.NDArray[np.float64],
    params_list: List[ParameterSet],
    regression_params: dict,
    num_domains: int,
    num_quality_criteria: int,
) -> Tuple[float, RegressionModel]:
    """Train a single CV fold and return (R², fold model).

    The fold model is returned so the caller can retain it for LCB uncertainty
    estimation (std of fold-aggregate z-sums). Previously only R² was returned
    and the model was discarded.
    """
    cv_val_idx = folds[fold_idx]
    cv_train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

    cv_train_params = [params_list[i] for i in cv_train_idx]
    cv_val_params = [params_list[i] for i in cv_val_idx]
    cv_train_losses = task_losses[cv_train_idx]
    cv_val_losses = task_losses[cv_val_idx]

    cv_model = RegressionModel(model_type="lightgbm", **regression_params)
    cv_model.fit(
        cv_train_params,
        cv_train_losses,
        num_domains=num_domains,
        num_criteria=num_quality_criteria,
        eval_params_list=cv_val_params,
        eval_losses=cv_val_losses,
        verbose=False,
    )

    return float(cv_model.score(cv_val_params, cv_val_losses)), cv_model


def train_single_task(
    task: str,
    task_losses: npt.NDArray[np.float64],
    params_list: List[ParameterSet],
    train_idx: npt.NDArray[np.int64],
    val_idx: npt.NDArray[np.int64],
    folds: Optional[List[npt.NDArray[np.int64]]],
    n_folds: int,
    regression_params: dict,
    num_domains: int,
    num_quality_criteria: int,
) -> Dict[str, Any]:
    """Train a single task's model (CV or single split). Returns result dict."""
    from joblib import Parallel, delayed

    result = {"task": task}

    result["train_stats"] = (
        float(np.mean(task_losses[train_idx])),
        float(np.std(task_losses[train_idx])),
    )

    if folds is not None and n_folds > 1:
        fold_results = Parallel(n_jobs=n_folds, prefer="threads")(
            delayed(train_single_task_cv_fold)(
                fold_idx=fold_idx,
                folds=folds,
                n_folds=n_folds,
                task_losses=task_losses,
                params_list=params_list,
                regression_params=regression_params,
                num_domains=num_domains,
                num_quality_criteria=num_quality_criteria,
            )
            for fold_idx in range(n_folds)
        )
        fold_r2s = [r for r, _ in fold_results]
        result["fold_models"] = [m for _, m in fold_results]
        result["r2"] = float(np.mean(fold_r2s))
    else:
        result["fold_models"] = []
        train_params = [params_list[i] for i in train_idx]
        val_params = [params_list[i] for i in val_idx] if len(val_idx) > 0 else None
        task_train_losses = task_losses[train_idx]
        task_val_losses = task_losses[val_idx] if len(val_idx) > 0 else None

        temp_model = RegressionModel(model_type="lightgbm", **regression_params)
        temp_model.fit(
            train_params,
            task_train_losses,
            num_domains=num_domains,
            num_criteria=num_quality_criteria,
            eval_params_list=val_params,
            eval_losses=task_val_losses,
            verbose=False,
        )

        if len(val_idx) > 0:
            result["r2"] = float(temp_model.score(val_params, task_val_losses))
        else:
            result["r2"] = float(temp_model.score(train_params, task_train_losses))

    train_params = [params_list[i] for i in train_idx]
    val_params = [params_list[i] for i in val_idx] if len(val_idx) > 0 else None
    task_train_losses = task_losses[train_idx]
    task_val_losses = task_losses[val_idx] if len(val_idx) > 0 else None

    model = RegressionModel(model_type="lightgbm", **regression_params)
    model.fit(
        train_params,
        task_train_losses,
        num_domains=num_domains,
        num_criteria=num_quality_criteria,
        eval_params_list=val_params,
        eval_losses=task_val_losses,
        verbose=False,
    )

    result["model"] = model
    result["train_r2"] = float(model.score(train_params, task_train_losses))

    return result


def bootstrap_one(
    seed: int,
    params_list: List[ParameterSet],
    losses: npt.NDArray[np.float64],
    n_features: int,
    num_domains: int,
    num_criteria: int,
    regression_params: dict,
) -> Optional[Tuple[float, RegressionModel]]:
    """One bootstrap sample → (oob R², model) or None if sample too small."""
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
            **regression_params,
        )
        model.fit(
            boot_train_params,
            boot_train_losses,
            num_domains=num_domains,
            num_criteria=num_criteria,
            eval_params_list=oob_params,
            eval_losses=oob_losses,
            verbose=False,
        )
        r2 = float(model.score(oob_params, oob_losses))
        return (r2, model)
    except Exception:
        return None
