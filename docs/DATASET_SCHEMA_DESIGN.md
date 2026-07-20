# QuaDMix DatasetSchema — 自定义数据集适配设计

> **版本**: v1.0  
> **日期**: 2026-07-20  
> **状态**: 设计完成，待实现

---

## 1. 背景与动机

QuaDMix 原始实现硬编码了 Essential-Web v1.0 的 schema（22 域、5 质量列、`domain`/`qs_*` 列名）。当用户使用其他数据集（如 STEM 数据集只有 4 域、6 质量列、`category_name` 域列）时，系统直接 crash：

```
ArrowInvalid: No match for FieldRef.Name(domain)
```

**目标**：让用户只需写一个 YAML 配置文件 + 提供符合最低要求的 parquet 数据，即可在任意数据集上运行 QuaDMix pipeline，零代码改动。

---

## 2. 核心设计

### 2.1 DatasetSchema

YAML 配置对象，将 parquet 列名映射到算法概念（domain / quality / text / char_count）。

```yaml
# schema_stem.yaml — STEM 数据集示例
domain_col: category_name
quality_cols:
  - category_score        # 默认 higher_better=true
  - stem_relevance
  - knowledge_value
  - notation_fidelity
  - rigor_coherence
  - noise_level           # 噪声越低越好
    higher_better: false
text_col: text
char_count_col: null      # null → 自动从 text 计算；也可填具体列名如 "doc_len"
```

```yaml
# 无 YAML 时 = Essential-Web 默认 schema（向后兼容）
domain_col: domain
quality_cols: [qs_dclm, qs_fineweb_edu_approx, qs_english, qs_eai_general_math, qs_eai_open_web_math]
text_col: text
char_count_col: doc_char_count
```

### 2.2 数据流

```
用户: YAML + parquet 目录
          ↓
DatasetSchema.from_yaml(path)        ← 解析 YAML，得到列映射配置
          ↓
ShardMetadataManager(dir, schema)    ← 按 schema 读取对应列
          ↓
  · domain_col 是 string → astype("category").cat.codes → int 0..M-1
  · domain_col 是非连续 int → remap 到 0..M-1
  · char_count_col=null + text_col 存在 → 从 len(str(text)) 计算，打印 WARNING
  · char_count_col=null + text_col 不存在 → 报错退出
          ↓
metadata_manager 暴露:
  · num_domains, num_quality_criteria   ← 自动检测
  · detected_domain_names               ← string 域的 unique 值 / int 域的 D0,D1,...
  · detected_quality_names              ← quality_cols 列名
  · domain_label_map                    ← {str: int} 映射（仅 string domain）
          ↓
QuaDMixConfig(M=mgr.num_domains, N=mgr.num_quality_criteria)  ← 动态填入，不再硬编码
          ↓
pipeline.run(..., domain_names=mgr.detected_domain_names,
              quality_names=mgr.detected_quality_names)
```

### 2.3 初始化顺序变更

**当前**（硬编码 22/5）：
```
QuaDMixConfig(num_domains=22, num_quality_criteria=5) → ShardMetadataManager → pipeline
```

**新**（动态检测）：
```
DatasetSchema → ShardMetadataManager(schema) → QuaDMixConfig(M=mgr.num_domains, N=mgr.num_quality_criteria) → pipeline
```

---

## 3. 数据集要求

### 3.1 必须满足（系统校验，不满足则报错）

| 要求 | 说明 | 报错行为 |
|---|---|---|
| Parquet 格式 | 目录下有 `*.parquet` 文件 | `FileNotFoundError: No .parquet files found in ...` |
| 1 个 domain 列 | 每行一个域标签，string 或 int dtype | `ValueError: 列 'category_name' 不存在。可用列: [text, score, ...]` |
| N 个 quality 列 | 全部 float dtype，不允许 NaN | `ValueError: 列 'stem_relevance' 不存在 / 不是 float dtype` |
| 所有 shard schema 一致 | 同样的列名和 dtype | `ValueError: shard 15 的 'category_name' dtype 是 int64，与 shard 0 的 object 不一致` |

### 3.2 char_count 处理逻辑

```
char_count_col 在 YAML 中指定为某列名:
  · parquet 中存在 → 正常读取
  · parquet 中不存在 → 报错: "列 'doc_len' 不存在。可用列: [...]"

char_count_col=null (YAML 中省略或设为 null):
  · text_col 存在 → 从 text 计算 len(str(t))
    打印 WARNING:
    "[ShardMetadataManager] 从 text 列计算 doc_char_count (XXX docs)。
     建议预处理时直接生成该列以加速后续加载。"
  · text_col 不存在 → 报错退出:
    "无法计算文档字符数: char_count_col 未指定且 text_col 不存在。
     请在 YAML 中指定 char_count_col 或 text_col。"

char_count_col=null + text_col 存在但部分文本为空:
  → 空文本的 char_count 设为 0（不使用 fallback 常量）
```

**不使用 fallback 常量（如 200）**。200 是不靠谱的数字，会误导 token 估算和采样决策。要么从数据计算，要么报错让用户自己处理。

### 3.3 quality 分数方向

| 情况 | 处理 |
|---|---|
| 默认（YAML 未标注） | **higher_better=true** — 分数越高，质量越好 |
| YAML 中标注 `higher_better: false` | 分数越低，质量越好（如 noise_level） |
| 算法内部 | percentile rank 0 = 最高质量。higher_better=true 时取 `1 - percentile`；higher_better=false 时取 `percentile` |

### 3.4 domain 列处理

| dtype | 处理 | domain_names |
|---|---|---|
| string (object) | `astype("category").cat.codes` → 0..M-1 | unique 值按 category 顺序：["数学", "化学", "生物学", "物理"] |
| 连续 int (0..M-1) | 直接使用 | ["D0", "D1", ..., "D{M-1}"] |
| 非连续 int (如 [1,5,10,20]) | remap 到 0..M-1 | ["D1→0", "D5→1", "D10→2", "D20→3"] 或用原始值 |

### 3.5 统计建议（不硬性校验，但打印 WARNING）

| 情况 | WARNING |
|---|---|
| 某 domain 文档数 < 100 | `"domain '物理' 仅 23 条数据 (0.1%)，该域的 percentile rank 和采样参数可能不稳定"` |
| quality 列含 NaN | `"列 'stem_relevance' 有 1.2% NaN 值，已填充为 0。建议在预处理时处理缺失值"` |
| 文本列有空字符串 > 5% | `"text 列有 8% 空文档，这些文档的 char_count=0，proxy 训练可能受影响"` |

---

## 4. 文件名要求（无要求）

**文件名不需要包含 shard index**。原始代码从文件名正则提取 `(\d+)` 作为 shard_idx，但实际数据访问全部用文件在 sorted 列表中的位置索引，shard_idx 仅用于日志展示。

新行为：
- 文件名有数字 → 解析出来用于日志（`"shard 12 loaded"`）
- 文件名无数字 → 用位置索引（`"shard #3 loaded"`）
- 不匹配 → 不报错，不阻塞

用户可以随意命名文件：`batch_a.parquet`, `stem_data.parquet`, `shard_000.parquet` 都行。

---

## 5. `row_in_shard` 列（可选）

原始 Essential-Web 预处理会在每个 parquet 中添加 `row_in_shard` 列，用于 `read_texts()` 的 parquet row filter 高效读取。

新行为：
- parquet 有 `row_in_shard` → 使用高效 row filter（只读需要的行）
- parquet 无 `row_in_shard` → 读整个 text 列，按位置选取（稍慢但功能正常）

用户不需要在预处理中添加此列。

---

## 6. YAML 格式详细说明

```yaml
# ── 必填 ──
domain_col: <string>            # 哪列是域标签（string 或 int dtype）
quality_cols:                   # 哪些列是质量分数（float dtype）
  - <col_name>                  # 默认 higher_better=true
  - name: <col_name>            # 显式指定方向
    higher_better: false        # 越低越好

# ── 半必填 ──
text_col: <string>              # 文本列名（proxy 实验需要；无 text → 只能做参数搜索）
char_count_col: <string|null>   # 字符数列名；null → 从 text 计算；无 text + null → 报错

# ── 可选 ──
domain_names: [<string>, ...]   # 覆盖自动检测的域名称（用于输出序列化）
quality_names: [<string>, ...]  # 覆自动检测的质量名（默认用 quality_cols 列名）
```

所有字段有默认值，不提供 YAML 时使用 Essential-Web 默认值：
- `domain_col: "domain"`
- `quality_cols: ["qs_dclm", "qs_fineweb_edu_approx", "qs_english", "qs_eai_general_math", "qs_eai_open_web_math"]`
- `text_col: "text"`
- `char_count_col: "doc_char_count"`

---

## 7. 用户操作流程

### 新数据集接入：3 步

```
Step 1: 把数据转成 parquet（每行一个样本，列是平铺的值，不是嵌套 struct/list）
        必须包含: 1 个 domain 列 + N 个 quality float 列
        强烈建议包含: 1 个 text 列（proxy 实验需要）

Step 2: 写一个 YAML 配置文件，指定列名映射

Step 3: 运行命令
        python scripts/runners/run_essential_web_v1.py \
            --schema schema_stem.yaml \
            --preprocessed-dir /path/to/shards \
            --quick
```

**零代码改动。** 系统自动处理：string domain→int、M/N 检测、char_count 计算、列名校验。

### 列名不匹配时的报错示例

用户忘了写 YAML，直接在 STEM 数据上运行：

```
ValueError: Schema 校验失败 — 以下列不存在于 parquet 中:
  - domain (缺失)
  - qs_dclm (缺失)
  - qs_fineweb_edu_approx (缺失)
  - qs_english (缺失)
  - qs_eai_general_math (缺失)
  - qs_eai_open_web_math (缺失)
  - doc_char_count (缺失)

可用列 (10):
  text (object), category_name (object), category_score (float64),
  stem_relevance (float64), knowledge_value (float64),
  notation_fidelity (float64), rigor_coherence (float64),
  noise_level (float64), source_file (object), source_record_idx (int64)

请创建 YAML 配置文件并使用 --schema 指定。
示例:
  domain_col: category_name
  quality_cols: [category_score, stem_relevance, ...]
  text_col: text
```

---

## 8. 实现计划

| Step | 文件 | 改动 |
|---|---|---|
| 1 | `src/quadmix/data/dataset_schema.py` | **新建** — DatasetSchema dataclass + YAML loader + _validate() |
| 2 | `src/quadmix/data/metadata_manager.py` | 改造 — 接受 schema 参数，动态读列，string→int，char_count 计算 |
| 3 | `scripts/runners/run_essential_web_v1.py` | 改造 — --schema CLI arg + swap 初始化顺序 + 动态 M/N |
| 4 | `src/quadmix/pipeline/essential_proxy_runner.py` | 去硬编码 — domain_names/quality_names 从外部传入 |
| 5 | `src/quadmix/pipeline/real_pipeline.py` | 小改 — domain_col/quality_cols 默认值从 schema 取 |
| 6 | 端到端测试 | Essential-Web 向后兼容 + STEM 数据集验证 |

### 向后兼容保证

- 无 `--schema` → 默认 DatasetSchema(Essential-Web) → 行为与当前完全一致
- 有 `--schema schema_stem.yaml` → STEM 配置 → M=4, N=6 自动检测
