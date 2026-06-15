# v4.2 Design: Per-Task Loss with Discrimination-Adaptive Weighting

**Version**: v4.2  
**Date**: 2026-06-12  
**Status**: Design Complete, Pending Implementation

---

## 1. Background

### 1.1 Problem Statement

QuaDMix 的最终目标：通过 1M proxy 模型搜索最优数据配比（域 × 质量 × 比例），使 1.3B 模型 mid-training 后在 22 个下游基准上得分最大化。

**当前 v4 的问题**：验证集使用单一 aggregate val_loss 作为 LightGBM 的 target。这导致：

| 问题 | 后果 |
|------|------|
| 高区分度任务的信号被低区分度任务稀释 | LightGBM 学不到明确信号 |
| 完全排除低区分度任务 | 搜索时不考虑该基准，1.3B 可能在该基准上表现差 |
| 所有任务等权平均 | 无法区分"强信号"和"弱信号"任务 |

### 1.2 Core Insight

**1M proxy 学的是文本分布，不是 Q-A 对应关系。**

- 验证集的每个任务代表一个**文本域探针**
- val_loss 衡量：训练数据的文本分布是否覆盖了该基准所需的文本域
- 不同任务的 loss 在不同实验间的**方差**反映了该任务的**区分度**
- 区分度高的任务，LightGBM 预测更准，应该给予更高权重

### 1.3 Design Goal

```
Per-task loss + 区分度自适应加权
  ├── 保留所有基准任务（不丢弃低区分度任务）
  ├── 高区分度任务权重高（信号强，预测准）
  ├── 低区分度任务权重低但不为零（保留覆盖）
  └── 零方差任务自动排除（无信号 = 纯噪声）
```

---

## 2. Design Overview

### 2.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Validation Set (v4)                          │
│  token_ids: [12371, 2048]                                       │
│  loss_mask: [12371, 2048]                                       │
│  task_labels: ['hellaswag_zeroshot', ..., 'commonsense_qa']     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              _run_validation (modified)                         │
│  返回: (aggregate_loss, per_task_losses)                        │
│  per_task_losses = {                                            │
│    'hellaswag_zeroshot': 3.42,                                  │
│    'lambada_openai': 4.15,                                      │
│    'arc_easy': 2.89,                                            │
│    ...                                                          │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ProxyResult (modified)                             │
│  validation_loss: float (aggregate, backward compatible)        │
│  per_task_losses: Dict[str, float] (new)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              QuaDMixOptimizer.train_regressor (modified)        │
│  1. 训练 aggregate 模型（向后兼容）                              │
│  2. 训练 per-task 模型（每个任务一个 LightGBM）                  │
│  3. 计算区分度权重（标准差归一化）                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              QuaDMixOptimizer.search_optimal (modified)         │
│  1. 用 per-task 模型预测每个任务的 loss                         │
│  2. 加权聚合: predicted_loss = Σ w_i * loss_i                   │
│  3. 搜索最小化 weighted_loss 的参数                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Backward Compatibility

| Validation Set | task_labels | Behavior |
|----------------|:-----------:|----------|
| openhermes-10k | ❌ | 只训练 aggregate 模型，行为与 v4.1 完全一致 |
| core_bmk_v4 | ✅ | 训练 per-task 模型 + 加权聚合搜索 |

---

## 3. Detailed Design

### 3.1 ProxyResult Dataclass

**File**: `src/quadmix/core/types.py`

```python
@dataclass
class ProxyResult:
    parameters: ParameterSet
    validation_loss: float  # aggregate loss（向后兼容）
    per_task_losses: Optional[Dict[str, float]] = None  # NEW: per-task losses
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Impact**: 
- 所有创建 ProxyResult 的地方需要检查是否需要填充 per_task_losses
- 错误处理路径（tokenize_failed, worker_crash）保持 per_task_losses=None

### 3.2 Validation Loading

**File**: `src/quadmix/pipeline/essential_proxy_runner.py`  
**Location**: `__init__` (line ~145)

```python
# Current
val_data = torch.load(self.val_data_path, map_location="cpu", weights_only=False)
self._val_token_ids = val_data["token_ids"]
self._val_loss_mask = val_data["loss_mask"]

# Modified
val_data = torch.load(self.val_data_path, map_location="cpu", weights_only=False)
self._val_token_ids = val_data["token_ids"]
self._val_loss_mask = val_data["loss_mask"]
self._val_task_labels = val_data.get("task_labels", None)  # NEW
if self._val_task_labels:
    unique_tasks = sorted(set(self._val_task_labels))
    print(f"[ProxyRunner] Val tasks: {len(unique_tasks)} tasks: {unique_tasks}")
else:
    print(f"[ProxyRunner] Val: no task_labels (aggregate-only mode)")
```

### 3.3 _run_validation

**File**: `src/quadmix/pipeline/essential_proxy_runner.py`  
**Location**: `_run_validation` (line ~1150)

```python
# Current signature
def _run_validation(self, model, device) -> float:

# Modified signature
def _run_validation(self, model, device) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Run validation on full validation set.
    
    Returns:
        Tuple of (aggregate_loss, per_task_losses)
        - aggregate_loss: mean loss across all documents
        - per_task_losses: dict of task_name -> mean loss (None if no task_labels)
    """
    import torch.nn.functional as F
    model.eval()
    bs = self.block_size
    val_n = len(self._val_token_ids)
    val_tokens = self._val_token_ids[:val_n, :bs].to(device)
    val_mask = self._val_loss_mask[:val_n, :bs].to(device)
    
    with torch.no_grad():
        val_bs = min(96, val_n)
        per_doc_losses = []
        for start in range(0, len(val_tokens), val_bs):
            end = min(start + val_bs, len(val_tokens))
            ids_in = val_tokens[start:end, :-1]
            ids_tgt = val_tokens[start:end, 1:]
            mask_tgt = val_mask[start:end, 1:]
            hidden = model(ids_in, return_hidden=True)
            loss = chunked_loss_per_token_from_hidden(model, hidden, ids_tgt, chunk_size=2048)
            assistant_count = mask_tgt.float().sum(dim=1).clamp(min=1)
            per_doc = (loss * mask_tgt.float()).sum(dim=1) / assistant_count
            per_doc_losses.append(per_doc)
            del hidden, loss, per_doc
        
        all_losses = torch.cat(per_doc_losses)
        aggregate_loss = float(all_losses.mean())
    
    # NEW: Compute per-task losses
    per_task_losses = None
    if self._val_task_labels:
        per_task_losses = {}
        for task in set(self._val_task_labels):
            task_indices = [i for i, t in enumerate(self._val_task_labels) if t == task]
            task_loss = float(all_losses[task_indices].mean())
            per_task_losses[task] = task_loss
    
    del val_tokens, val_mask, per_doc_losses, all_losses
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    model.train()
    
    return aggregate_loss, per_task_losses
```

### 3.4 Checkpoint Validation

**File**: `src/quadmix/pipeline/essential_proxy_runner.py`  
**Location**: training loop checkpoint (line ~1030)

```python
# Current
ckpt_val = self._run_validation(model, device)
self._ckpt_results[step_ct] = ckpt_val

# Modified
ckpt_val, _ = self._run_validation(model, device)  # ignore per-task for checkpoints
self._ckpt_results[step_ct] = ckpt_val
```

**Rationale**: Checkpoint validation 用于监控训练进度，per-task 信息不必要，保持简单。

### 3.5 Final Validation & ProxyResult

**File**: `src/quadmix/pipeline/essential_proxy_runner.py`  
**Location**: end of `_train_one` (line ~1060)

```python
# Current
val_loss = self._run_validation(model, device)
# ...
return ProxyResult(parameters=params, validation_loss=val_loss, metadata=meta)

# Modified
val_loss, per_task_losses = self._run_validation(model, device)
# ...
meta["per_task_losses"] = per_task_losses  # NEW: save to metadata
# ...
return ProxyResult(
    parameters=params,
    validation_loss=val_loss,
    per_task_losses=per_task_losses,  # NEW
    metadata=meta
)
```

### 3.6 revalidate_from_saved

**File**: `src/quadmix/pipeline/essential_proxy_runner.py`  
**Location**: `revalidate_from_saved` (line ~1181)

```python
# Current
def revalidate_from_saved(self, model_path: str, device_type: str = "cpu") -> float:
    # ...
    val_loss = self._run_validation(model, device)
    # ...
    return val_loss

# Modified
def revalidate_from_saved(
    self, model_path: str, device_type: str = "cpu"
) -> Tuple[float, Optional[Dict[str, float]]]:
    # ...
    val_loss, per_task_losses = self._run_validation(model, device)
    # ...
    return val_loss, per_task_losses
```

### 3.7 QuaDMixOptimizer

**File**: `src/quadmix/pipeline/optimizer.py`

#### 3.7.1 New Attributes

```python
class QuaDMixOptimizer:
    def __init__(self, config, regression_params=None):
        # ... existing ...
        self._task_models: Optional[Dict[str, RegressionModel]] = None
        self._task_weights: Optional[Dict[str, float]] = None
        self._task_r2: Optional[Dict[str, float]] = None
```

#### 3.7.2 train_regressor

```python
def train_regressor(self) -> RegressionModel:
    # ... existing aggregate model training ...
    
    # NEW: Check for per-task losses
    has_per_task = all(r.per_task_losses for r in self._proxy_results)
    
    if has_per_task:
        self._train_per_task_models(train_params, train_losses, val_params, val_losses, train_idx, val_idx)
    
    return self._regressor

def _train_per_task_models(self, train_params, train_losses, val_params, val_losses, train_idx, val_idx):
    """Train per-task LightGBM models and compute discrimination-adaptive weights."""
    
    # Get task names from first result
    tasks = [t for t in self._proxy_results[0].per_task_losses.keys()]
    
    # Prepare per-task loss arrays
    all_task_losses = {task: np.array([r.per_task_losses[task] for r in self._proxy_results])
                     for task in tasks}
    
    # Filter out zero-variance tasks
    task_variances = {task: np.var(losses) for task, losses in all_task_losses.items()}
    valid_tasks = {task: var for task, var in task_variances.items() if var > 1e-8}
    
    if len(valid_tasks) < len(task_variances):
        excluded = set(task_variances.keys()) - set(valid_tasks.keys())
        print(f"[QuaDMixOptimizer] WARNING: Excluding {len(excluded)} zero-variance tasks: {excluded}")
    
    if len(valid_tasks) == 0:
        print("[QuaDMixOptimizer] ERROR: All tasks have zero variance, skipping per-task models")
        return
    
    # Train per-task models
    self._task_models = {}
    self._task_r2 = {}
    
    print(f"[QuaDMixOptimizer] Training {len(valid_tasks)} per-task models...")
    
    for task in valid_tasks:
        task_losses = all_task_losses[task]
        task_train_losses = task_losses[train_idx]
        task_val_losses = task_losses[val_idx] if len(val_idx) > 0 else None
        
        model = RegressionModel(model_type="lightgbm", **self.regression_params)
        model.fit(
            train_params,
            task_train_losses,
            num_domains=self.config.num_domains,
            num_criteria=self.config.num_quality_criteria,
            eval_params_list=val_params if len(val_idx) > 0 else None,
            eval_losses=task_val_losses,
        )
        
        self._task_models[task] = model
        
        # Compute per-task R²
        if len(val_idx) > 0:
            task_r2 = model.score(val_params, task_val_losses)
            self._task_r2[task] = task_r2
        else:
            task_r2 = model.score(train_params, task_train_losses)
            self._task_r2[task] = task_r2
    
    # Compute discrimination-adaptive weights (standard deviation normalization)
    task_stds = {task: np.sqrt(var) for task, var in valid_tasks.items()}
    total_std = sum(task_stds.values())
    self._task_weights = {task: std / total_std for task, std in task_stds.items()}
    
    # Print summary
    print(f"[QuaDMixOptimizer] Per-task models trained:")
    print(f"  {'Task':<30} {'R²':>8} {'Weight':>8} {'Std':>8}")
    print(f"  {'-'*54}")
    for task in sorted(valid_tasks.keys(), key=lambda t: -self._task_weights[t]):
        r2 = self._task_r2[task]
        weight = self._task_weights[task]
        std = task_stds[task]
        print(f"  {task:<30} {r2:>8.4f} {weight:>8.4f} {std:>8.4f}")
```

#### 3.7.3 search_optimal

```python
def search_optimal(self, n_search_points=None, top_k=None):
    # ... existing candidate generation ...
    
    if self._task_models and self._task_weights:
        # Per-task prediction + weighted aggregation
        task_preds = {}
        for task, model in self._task_models.items():
            task_preds[task] = model.predict(candidates)
        
        predicted_losses = sum(
            self._task_weights[task] * task_preds[task]
            for task in task_preds
        )
        print(f"[QuaDMixOptimizer] Search: per-task weighted prediction ({len(self._task_models)} tasks)")
    elif hasattr(self, '_bootstrap_models') and len(self._bootstrap_models) > 0:
        # Fallback to bootstrap ensemble
        all_preds = np.array([m.predict(candidates) for m in self._bootstrap_models])
        predicted_losses = np.mean(all_preds, axis=0)
        print(f"[QuaDMixOptimizer] Search: bootstrap ensemble ({len(self._bootstrap_models)} models)")
    else:
        # Fallback to aggregate model
        predicted_losses = self._regressor.predict(candidates)
        print(f"[QuaDMixOptimizer] Search: aggregate model")
    
    # ... existing top-K averaging ...
```

### 3.8 reval_with_new_valset.py

**File**: `scripts/runners/reval_with_new_valset.py`  
**Location**: main revalidation loop (line ~247)

```python
# Current
new_val_loss = runner.revalidate_from_saved(model_path, device_type=args.device_type)

# Modified
new_val_loss, new_per_task_losses = runner.revalidate_from_saved(
    model_path, device_type=args.device_type
)

# Update metadata
new_meta["per_task_losses"] = new_per_task_losses

# Update ProxyResult
results.append(ProxyResult(
    parameters=params,
    validation_loss=new_val_loss,
    per_task_losses=new_per_task_losses,  # NEW
    metadata=new_meta,
))
```

### 3.9 report.py (Optional Enhancement)

**File**: `src/quadmix/pipeline/report.py`

可以在 Table 1 中增加 per-task loss 列（可选），但为了简洁，建议保持 aggregate val_loss，在单独的 section 中报告 per-task 统计。

---

## 4. Weight Calculation: Standard Deviation Normalization

### 4.1 Why Standard Deviation, Not Variance?

| Normalization | Task A (var=0.10) | Task B (var=0.01) | Weight Ratio |
|---------------|:-----------------:|:-----------------:|:------------:|
| **Variance** | 0.91 | 0.09 | 10:1 |
| **Std Dev** | 0.76 | 0.24 | 3.16:1 |

**Rationale**:
- Variance normalization 让高区分度任务完全主导，低区分度任务几乎被忽略
- Standard deviation 更平衡：高区分度任务权重更高，但低区分度任务仍有影响力
- 符合"所有下游基准都要考虑"的设计目标

### 4.2 Zero-Variance Handling

**Strategy**: Exclude + Warning

```python
# Filter zero-variance tasks
valid_tasks = {task: var for task, var in task_variances.items() if var > 1e-8}

if len(valid_tasks) < len(task_variances):
    excluded = set(task_variances.keys()) - set(valid_tasks.keys())
    print(f"[QuaDMixOptimizer] WARNING: Excluding {len(excluded)} zero-variance tasks: {excluded}")

if len(valid_tasks) == 0:
    print("[QuaDMixOptimizer] ERROR: All tasks have zero variance, skipping per-task models")
    self._task_models = None
    self._task_weights = None
```

**Rationale**:
- Zero variance = 该任务在所有实验中 loss 完全相同 = 无区分度信号
- 排除后，其他任务的权重自动重新归一化
- 打印警告，让用户知道发生了什么

---

## 5. R² Evaluation Strategy

### 5.1 What to Report

| Metric | Purpose | Use for Model Selection? |
|--------|---------|:------------------------:|
| Aggregate R² | Overall model quality | ✅ Yes |
| Per-task R² | Diagnostic: which tasks are well-predicted | ❌ No |

### 5.2 Implementation

```python
# After training per-task models
print(f"[QuaDMixOptimizer] Per-task R² (diagnostic):")
for task in sorted(self._task_r2.keys(), key=lambda t: -self._task_r2[t]):
    r2 = self._task_r2[task]
    weight = self._task_weights[task]
    print(f"  {task:<30} R²={r2:.4f}  weight={weight:.4f}")
```

**Note**: Aggregate R² 仍然用于 bootstrap CI 和模型选择，per-task R² 仅用于诊断。

---

## 6. Files to Modify

| File | Changes | Lines Affected |
|------|---------|----------------|
| `src/quadmix/core/types.py` | ProxyResult add `per_task_losses` | ~250 |
| `src/quadmix/pipeline/essential_proxy_runner.py` | 1. Load task_labels<br>2. `_run_validation` return per-task<br>3. Checkpoint handling<br>4. Final validation<br>5. `revalidate_from_saved`<br>6. `save_summary` | ~145, ~1030, ~1060, ~1148, ~1181, ~1216 |
| `src/quadmix/pipeline/optimizer.py` | 1. New attributes<br>2. `_train_per_task_models`<br>3. `search_optimal` weighted prediction | ~285, ~375 (new), ~467 |
| `scripts/runners/reval_with_new_valset.py` | Handle per-task losses in revalidation | ~247 |

**Files NOT modified**:
- `data/openhermes_10k_assistant_tokenized.pt` (no task_labels, backward compatible)
- `data/core_bmk_10tasks_v4_tokenized.pt` (already has task_labels)
- `src/quadmix/pipeline/real_pipeline.py` (uses `r.validation_loss`, no change needed)
- `src/quadmix/pipeline/report.py` (uses `meta["val_loss"]`, no change needed)

---

## 7. Testing Plan

### 7.1 Unit Tests

1. **ProxyResult backward compatibility**
   - Create ProxyResult without per_task_losses → should work
   - Create ProxyResult with per_task_losses → should work

2. **_run_validation with task_labels**
   - Load openhermes-10k (no task_labels) → returns (loss, None)
   - Load core_bmk_v4 (with task_labels) → returns (loss, dict)

3. **Weight calculation**
   - Normal case: 3 tasks with different variances → weights sum to 1
   - Zero-variance case: 1 task with var=0 → excluded, others re-normalized
   - All zero-variance: → task_models=None, fallback to aggregate

4. **search_optimal**
   - With per-task models → uses weighted prediction
   - Without per-task models → uses aggregate model
   - With bootstrap ensemble → uses ensemble (takes precedence)

### 7.2 Integration Tests

1. **Run 10 experiments with core_bmk_v4**
   - Verify per-task losses are recorded in meta.json
   - Verify per-task models are trained
   - Verify weights are computed and printed

2. **Run search_optimal**
   - Verify weighted prediction is used
   - Verify optimal parameters are reasonable

3. **Run with openhermes-10k**
   - Verify behavior is identical to v4.1 (no per-task models)

---

## 8. Migration Guide

### 8.1 For Existing Experiments

Existing experiment results (meta.json) do not have `per_task_losses`. The optimizer will:
1. Check `all(r.per_task_losses for r in self._proxy_results)`
2. If False → skip per-task training, use aggregate model only
3. Behavior is identical to v4.1

### 8.2 For New Experiments

New experiments with core_bmk_v4 will automatically:
1. Record per-task losses in meta.json
2. Train per-task models in optimizer
3. Use weighted prediction in search

---

## 9. Future Enhancements

### 9.1 Custom Weight Override

Allow users to specify custom weights via config:

```python
config = QuaDMixConfig(
    ...
    task_weight_override={
        'hellaswag_zeroshot': 0.3,
        'arc_easy': 0.2,
        ...
    }
)
```

### 9.2 Dynamic Weight Update

Update weights during search based on predicted loss variance (not just training loss variance).

### 9.3 Multi-Objective Optimization

Instead of weighted sum, use Pareto optimization to find parameters that are optimal across all tasks.

---

## 10. Summary

| Aspect | v4.1 | v4.2 |
|--------|------|------|
| Validation target | Single aggregate loss | Per-task losses |
| LightGBM models | 1 aggregate model | 1 aggregate + N per-task models |
| Search objective | Minimize aggregate loss | Minimize weighted sum of per-task losses |
| Weight calculation | Equal (1/N) | Discrimination-adaptive (std normalization) |
| Zero-variance handling | N/A | Exclude + warning |
| Backward compatibility | N/A | ✅ Fully compatible with openhermes-10k |
