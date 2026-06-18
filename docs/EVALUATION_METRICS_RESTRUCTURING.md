# QuaDMix 评估指标体系重构：从 R² 到 Spearman

> **版本**: v1  
> **日期**: 2026-06-18  
> **验证集**: core_bmk_21tasks_v5 (31,547 samples)  
> **实验数**: 874 experiments  
> **结果**: Overall R²=0.3154, Equal-Wt R²=0.4062, Spearman=待验证, Top-K Recall=待验证

---

## 1. 起因：一个令人困惑的数字

v5 验证集 revalidate 完成（874 实验），结果：

```
Overall Val R² = 0.3154
```

但 per-task CV R² 全部 > 0.2，加权平均 ≈ 0.59。21 个 task 全部 active，最好的 task R² 达到 0.84。

**问题**：为什么每个 task 单独预测都不错，组合起来 R² 只有 0.32？

## 2. 第一轮分析：Overall R² 的计算方式

当时的 Overall R² 计算逻辑（z-score 空间，R² 加权）：

```python
z_ensemble_pred  = Σ(R²ᵢ · z_predᵢ) / Σ(R²ᵢ)
z_ensemble_actual = Σ(R²ᵢ · z_actualᵢ) / Σ(R²ᵢ)
Overall R² = R²(z_ensemble_actual, z_ensemble_pred)
```

这是**组合的 R²**，不是 **R² 的组合**。数学上：

```
R²(Σ wᵢXᵢ, Σ wᵢYᵢ) ≠ Σ wᵢ · R²(Xᵢ, Yᵢ)
```

per-task R² 加权平均 ≈ 0.59，但 ensemble R² = 0.32。差距来自哪里？

## 3. 第二轮分析：误差相关性

关键发现：per-task 预测误差在加权求和时**叠加放大**而非互相抵消。

```
Ensemble error = Σ wᵢ · (z_predᵢ - z_actualᵢ)
```

21 个 LightGBM 模型都在同一组实验参数上训练，学到的 pattern 有相似性。对同一个实验，要么都偏高估，要么都偏低估 → 误差正相关 → 加权求和后误差叠加 → Overall R² 被拉低。

**这不是搜索策略的问题**。不管用等权还是 R² 加权，只要误差正相关，组合后的 R² 都会比单个 task R² 的平均值低。

## 4. 第三轮分析：搜索目标 vs 评判指标的一致性

核心逻辑链：

```
下游目标：min (1/21) Σ raw_lossᵢ(θ)          ← 21 个 benchmark 等权
搜索策略：min Σ R²ᵢ · pred_lossᵢ(θ)          ← R² 加权，降低噪声 task 影响
```

**搜索策略和评判指标必须一致**。评判指标应该直接衡量搜索目标的预测准确度：

```
评判指标 = R²(Σ wᵢ · actual_lossᵢ, Σ wᵢ · pred_lossᵢ)
```

### 4.1 空间选择：z-score vs raw

不同 task 的 loss 范围和分布差异很大（bigbench_cs_algorithms mean=11.6，lambada mean=5.4）。如果用 raw loss，大 loss task 自然有更高影响力。z-score 归一化消除这个差异，让不同 task 在同一尺度上比较。

**结论**：保持 z-score 空间 + R² 加权作为搜索策略和对应评判指标。

### 4.2 为什么 R² 加权搜索能帮到等权目标？

等权搜索 `min (1/21) Σ pred_lossᵢ(θ)` 的问题：低 R² task 的 `pred_loss` 方差大，搜索会被噪声带偏，找到一组"看起来好但实际差"的参数。R² 加权搜索降低噪声 task 的话语权，让搜索方向更稳定。

**本质是 bias-variance tradeoff**：
- 等权搜索：无偏但高方差（被噪声带偏）
- R² 加权搜索：有偏（忽略低 R² task 的真实信号）但低方差（方向更稳）

## 5. 关键发现：R² 不是搜索质量的最佳度量

### 5.1 loss 区分度问题

v5 数据质量修复后，aggregate loss std 从 0.164 塌缩到 0.049：

| 指标 | v4.3 (674 exp) | v5 (874 exp) |
|------|---------------|-------------|
| Aggregate loss std | 0.1636 | **0.0488** |
| Overall R² | 0.4586 | 0.3154 |

所有实验的 aggregate loss 挤在 6.17-6.48 的窄带里。预测误差只要 0.03 就和信号差不多大 → R² 必然低。

### 5.2 单 task R² 为什么高？

单 task 的 loss 方差远大于 aggregate：

```
Per-task loss std:
  bigbench_cs_algorithms: 0.66  ← 差异大，信号强 → R²=0.56
  lambada:              0.13  ← 差异中等 → R²=0.84
  boolq:                0.06

Aggregate loss std:     0.049  ← 21 个 task 平均后差异被抹平
```

21 个 task 平均时，有的 task loss 高、有的低，正负抵消 → aggregate loss 波动很小 → R² 低。

**类比**：问"张三数学好不好"能准确预测（单科差异大），问"张三 21 科综合成绩排名"难以区分（平均后大家分数差不多）。

### 5.3 R² 的局限性

R² 衡量的是**绝对值预测准确度**。但搜索只需要**排序正确** — 不需要预测值精确，只需要好参数排在差参数前面。

loss 区分度小时，R² 必然低，但排序可能仍然有效。

## 6. 最终设计：4 指标体系

### 6.1 指标定义

| 指标 | 公式 | 空间 | 权重 | 衡量什么 |
|------|------|------|------|---------|
| Overall R² | R²(Σ wᵢ·z_predᵢ, Σ wᵢ·z_actualᵢ) | z-score | R² 加权 | 搜索目标预测准确度 |
| Equal-Wt R² | R²((1/K)Σ raw_predᵢ, (1/K)Σ raw_actualᵢ) | raw | 等权 | 下游目标预测准确度 |
| Spearman | corr(rank(pred), rank(actual)) | rank | 等权 | 排序能力 |
| Top-K Recall | \|pred_top_k ∩ actual_top_k\| / k | rank | 等权 | 搜索命中率 |

### 6.2 设计逻辑

**Overall R²**：与搜索策略完全一致（z-score 空间 + R² 加权）。高值说明搜索目标的预测准确，搜索选出的 top-K 参数可靠。

**Equal-Wt R²**：与下游目标完全一致（raw space + 等权）。诊断指标，反映终极目标的预测质量。如果 Overall R² 高但 Equal-Wt R² 低，说明 R² 加权策略在牺牲低 R² task。

**Spearman**：搜索本质是排序选 top，不需要绝对值准。R² 可以低（loss 区分度小），但 Spearman 可能仍然高（排序正确）。这是搜索质量最直接的度量。

**Top-K Recall**：搜索选出的 top-K 参数，在真实排序中排第几。最实际的指标 — 直接回答"我选出的参数到底好不好"。

### 6.3 为什么 Spearman 和 Top-K 更合适

```
优化目标：argmin_θ (1/21) Σ raw_lossᵢ(θ)
```

- R² 衡量"预测值有多准" → loss 区分度小时必然低
- Spearman 衡量"排序对不对" → 不受绝对值影响
- Top-K 衡量"头部选对了没" → 搜索只关心头部

### 6.4 解读标准

| 指标 | Excellent | Good | Moderate | Weak |
|------|-----------|------|----------|------|
| Overall R² | > 0.6 | > 0.3 | — | < 0.3 |
| Equal-Wt R² | > 0.6 | > 0.3 | — | < 0.3 |
| Spearman | > 0.7 | > 0.5 | > 0.3 | < 0.3 |
| Top-K Recall | > 0.7 | > 0.5 | > 0.3 | < 0.3 |

### 6.5 诊断逻辑

- Spearman > 0.5 → 排序可靠，搜索结果可信（即使 R² < 0.5）
- Top-K Recall > 0.5 → 搜索找到了好参数
- Overall R² 高 + Equal-Wt R² 低 → R² 加权策略有效但可能牺牲低 R² task
- Overall R² 低 + Equal-Wt R² 高 → 考虑改用等权搜索
- 两者都低但 Spearman 高 → R² 低是因为 loss 区分度小，排序仍然有效

## 7. 经验总结

1. **R² 不是万能的**：当目标变量方差很小时，R² 必然低，但不代表模型无用。搜索场景下排序能力比绝对值预测更重要。

2. **评判指标必须和搜索目标一致**：搜索在什么空间、用什么权重，评判就在同一空间、同一权重下计算。否则数值无法解释搜索质量。

3. **多指标互补**：R² 衡量绝对值准确度，Spearman 衡量排序能力，Top-K 衡量头部命中率。三者从不同角度评估模型质量，组合使用才能全面判断。

4. **数据质量修复的副作用**：v5 验证集修复了 7 个数据质量问题，数据更干净，但 aggregate loss 区分度随之消失。这是一个 tradeoff — 数据更准确但信号更弱。
