# FDC Domain Mapping 设计：从 100 类到 23 类

> 2026-06-29 | 基于 Essential-Web v1.0 FDC 分类体系的领域映射方案

## 1. 背景

### 1.1 当前状态

QuaDMix 当前使用 Essential-Web v1.0 的 FDC (Free Decimal Correspondence) L1 分类，共 **10 个领域**（Dewey Decimal 体系）：

| ID | L1 Domain | FDC Code |
|----|-----------|----------|
| 0 | Industrial arts, Technology, and Engineering | 6xx |
| 1 | Social sciences | 3xx |
| 2 | Science and Natural history | 5xx |
| 3 | Religion | 2xx |
| 4 | Philology; or, Language and languages | 4xx |
| 5 | Literature | 8xx |
| 6 | History and Geography | 9xx |
| 7 | General works, books and libraries, information sciences | 0xx |
| 8 | Philosophy and psychology | 1xx |
| 9 | Arts | 7xx |

**问题**：10 类粒度太粗。例如 "Science" (5xx) 将数学、物理、生物、医学合为一类，但这些领域对下游任务的影响差异极大。

### 1.2 FDC 层级结构

FDC 是 3 层 Dewey Decimal 改编体系：

- **L1**：10 类（百位数字，已实现）
- **L2**：理论 100 类（十位数字前缀，如 51x=Mathematics, 53x=Physics）
- **L3**：~1000 类（三位数字，如 512=Algebra, 515=Analysis）

L2 标签存在于 `eai_taxonomy.free_decimal_correspondence.primary.labels.level_2`，但：
- 部分 L2 标签为空字符串（如 FDC code `508`, `700.0`）
- 标签名与标准 Dewey 不完全一致（FDC 是改编版）
- 预估实际有效 L2 约 70-90 个

### 1.3 目标

将 ~100 个 FDC L2 映射到 **~23 个领域**，对齐业界经验证的分类粒度。

**设计动机**（2026-06-29 更新）：L2 映射的核心价值不是改变域比例，而是**提高域内同质性**。当前 FDC L1 的 10 域太粗（如 Industrial 41% 内部混杂工程、医学、商业），DCLM 质量分在异质域内区分度低。拆分后每个 L2 子域更同质，质量筛选更精准。

---

## 2. 参考体系分析

### 2.1 NVIDIA Domain Classifier（26 类）

基于 Google Cloud NLP API 的扁平分类体系，DeBERTa V3 Base 模型，0.987 PR-AUC。已在 Nemotron-4 的 8T tokens 预训练中验证。

| # | Domain | # | Domain |
|---|--------|---|--------|
| 1 | Adult | 14 | Internet_and_Telecom |
| 2 | Arts_and_Entertainment | 15 | Jobs_and_Education |
| 3 | Autos_and_Vehicles | 16 | Law_and_Government |
| 4 | Beauty_and_Fitness | 17 | News |
| 5 | Books_and_Literature | 18 | Online_Communities |
| 6 | Business_and_Industrial | 19 | People_and_Society |
| 7 | Computers_and_Electronics | 20 | Pets_and_Animals |
| 8 | Finance | 21 | Real_Estate |
| 9 | Food_and_Drink | 22 | Science |
| 10 | Games | 23 | Sensitive_Subjects |
| 11 | Health | 24 | Shopping |
| 12 | Hobbies_and_Leisure | 25 | Sports |
| 13 | Home_and_Garden | 26 | Travel_and_Transportation |

**特点**：
- 扁平结构，无层级
- 面向 web 内容分类（含 Adult, Shopping 等 web-native 类别）
- Science 合为一类（粒度太粗）
- 有现成模型可直接使用

### 2.2 ClimbMix / Nemotron-CLIMB（21 类）

自动聚类方案（stella_en_400M_v5 embedding + k-means + GPT-4o 标注），NeurIPS 2025。

| Cluster | Topics |
|---------|--------|
| C1 | Environment, Public Health, Policy Development, Medical Innovation |
| C2 | Technology, Neurophysiology, Health and Safety, Innovative Research |
| C3 | Restoration Efforts, Climate and Ecosystem, Community Engagement |
| C4 | Diagnostics, Diseases, Prevention and Control |
| C5 | Vehicles, Ecology, Community, Conservation Efforts |
| C6 | Energy, Science, Materials, Nanostructures, Quantum Computing |
| C7 | Physics, Accelerators, Materials, Architecture, System |
| C8 | Biology, Genetics, Astronomy, Climate Science |
| C9 | Earth Sciences, Space Science, Scientific Collaboration |
| C10 | Health, Symptoms, Treatment, Therapy, Disorders, Conditions |
| C11 | Communication, Biography, History, Society, Policy |
| C12 | Culture, Education, Sustainability, Community, Public Health, Crime, Economy |
| C13 | Arts, Literature, Education, History |
| C14 | Geography, Government, Organization, Religion, Agriculture, Economy, Civilizations |
| C15 | Science, Technology, Education, Engineering, Collaboration |
| C16 | Science, Health, Minerals, Population, Agriculture, Vaccination, Welfare, Management |
| C17 | Role-Playing, Problem Solving, Mathematics, Algorithms |
| C18 | Revolution, Parliament, Efficiency, Communication, Animal Behavior |
| C19 | History, Culture, Economy, Energy, Market, Policy |
| C20 | Python, Code |
| C21 | Government, Law, Scientific Revolution, Music, Literature |

**特点**：
- 数据驱动，非预定义
- Cluster 跨域混合（如 C1 = Environment + Health + Policy）
- Science 拆成 4 个 cluster (C6-C9)，验证了细粒度拆分的合理性
- Code 独立成类 (C20)

### 2.3 两个体系的共识

| 维度 | NVIDIA | ClimbMix | 启示 |
|------|--------|----------|------|
| 总类别数 | 26 | 21 | **20-26 类是经验合理粒度** |
| Science | 合1 | 拆4 | ClimbMix 验证了拆分价值 |
| Code/CS | 独立 | 独立 | 必须独立 |
| Health/Medicine | 独立 | 拆2 | 至少独立1类 |
| Arts | 合1 | 合1 | 可合 |
| Sports | 独立 | — | 可独立 |

---

## 3. FDC-23 映射方案

### 3.1 设计原则

| 原则 | 说明 |
|------|------|
| **高价值域独立** | CS、数学、物理、生物、医学、工程、法律 — 对下游任务影响大且词汇差异大 |
| **同质域合并** | 宗教 10 合 1、文学 10 合 1、非英语语言合并 |
| **小数据域合并** | 手稿(09x)、形而上学(11x)等 web 上极少 |
| **对齐 NVIDIA 命名** | 尽量复用 NVIDIA 26 类的命名，便于对照 |
| **基于 FDC code 前缀映射** | 比文本标签更可靠（level_2 标签名不统一且可能为空） |

### 3.2 完整映射表

| ID | 名称 | FDC 前缀 | 合并 L2 数 | 对标 NVIDIA | 对标 ClimbMix |
|----|------|----------|-----------|-------------|---------------|
| 0 | **Computers_and_Electronics** | 00x | 1 | Computers_and_Electronics | C20 |
| 1 | **News_and_General_Works** | 01x-09x | 9 | News | — |
| 2 | **Philosophy_and_Psychology** | 1xx | 10 | People_and_Society(部分) | C13(部分) |
| 3 | **Religion** | 2xx | 10 | People_and_Society(部分) | C14(部分) |
| 4 | **Law_and_Government** | 32x, 34x | 2 | Law_and_Government | C18, C21 |
| 5 | **Economics_and_Finance** | 33x | 1 | Finance | C19(部分) |
| 6 | **Education** | 37x | 1 | Jobs_and_Education | C15(部分) |
| 7 | **People_and_Society** | 30x-31x, 35x-36x, 38x-39x, 92x | 7 | People_and_Society | C11, C12 |
| 8 | **English_Language** | 40x-42x | 3 | Books_and_Literature(部分) | — |
| 9 | **Other_Languages** | 43x-49x | 7 | Books_and_Literature(部分) | — |
| 10 | **Mathematics** | 51x | 1 | Science(部分) | C17 |
| 11 | **Physics_and_Chemistry** | 53x-54x | 2 | Science(部分) | C6, C7 |
| 12 | **Earth_and_Life_Sciences** | 50x, 52x, 55x-59x | 7 | Science(部分) | C8, C9 |
| 13 | **Medicine_and_Health** | 61x | 1 | Health | C1, C4, C10 |
| 14 | **Business_and_Management** | 65x | 1 | Business_and_Industrial(部分) | — |
| 15 | **Engineering** | 60x, 62x, 66x-69x | 6 | Business_and_Industrial(部分) | C15 |
| 16 | **Agriculture** | 63x | 1 | Food_and_Drink(部分) | C3, C5(部分) |
| 17 | **Arts_and_Entertainment** | 70x-78x | 9 | Arts_and_Entertainment | C13, C21 |
| 18 | **Sports_and_Recreation** | 79x | 1 | Sports | — |
| 19 | **Books_and_Literature** | 8xx | 10 | Books_and_Literature | C13(部分) |
| 20 | **History** | 90x, 93x-99x | 8 | People_and_Society(部分) | C14, C19 |
| 21 | **Geography_and_Travel** | 91x | 1 | Travel_and_Transportation | C14(部分) |
| 22 | **Home_Economics** | 64x | 1 | Home_and_Garden(部分) | — |

**总计：23 类**（全部有效领域，无 Other 类）

> **决策**：原 Other 类（FDC code=-1，8,462 docs，0.02%）数据质量差（avg tok/doc=14,447），直接丢弃。

### 3.3 关键决策说明

| 决策 | 理由 |
|------|------|
| Science 拆 4 类 (10-13) | NVIDIA 合1 太粗；ClimbMix 拆成 C6-C9 四个 cluster，验证了拆分的合理性 |
| Philosophy + Psychology 合1 (ID 2) | Web 上哲学数据极少，单独成类会近乎空；NVIDIA 无对应独立类 |
| Religion 独立 (ID 3) | 与 Philosophy 分离，web 上宗教数据量尚可 |
| People_and_Society 吸收 Biography (92x) | 对齐 NVIDIA 的 People_and_Society 范畴 |
| Business 拆 Management (65x) + Engineering (60x/62x/66-69x) | NVIDIA 合为 Business_and_Industrial 太粗，工程 vs 管理词汇差异大 |
| Arts 合 9 个 L2 (ID 17) | 对齐 NVIDIA 的 Arts_and_Entertainment，web 上艺术类同质性高 |
| English vs Other Languages 分离 (8, 9) | Web 数据以英语为主，非英语语言对 multilingual 能力有独立贡献 |
| History 从 Geography 分离 (20, 21) | 对齐 NVIDIA 的 Travel_and_Transportation 独立 |
| **Other 类直接丢弃** | FDC code=-1 的文档仅 8,462 条（0.02%），且 avg tok/doc=14,447（异常高），数据质量差。直接丢弃而非归入 Other 类，避免引入噪声和无效搜索维度 |
| **Home_Economics 独立成域 (ID 23)** | 64x（家政学/消费者科学）占 11.6%，与 People_and_Society 内容风格差异大（食谱 vs 社会学）。独立成域提高同质性，使 DCLM 质量分更有区分度 |
| **60x/66x-69x 归入 Engineering (ID 15)** | 化工、制造、建筑等工业技术与工程设计语义一致，合并后 Engineering 从 8.4% 升至 13.3%，可接受 |

### 3.4 FDC 无法覆盖的 NVIDIA 类别

以下 NVIDIA 类别是 web-native 分类，Dewey Decimal 体系无法映射：

| NVIDIA 类别 | 说明 |
|-------------|------|
| Adult | 成人内容 |
| Autos_and_Vehicles | 汽车（现代消费品） |
| Beauty_and_Fitness | 美容健身 |
| Pets_and_Animals | 宠物（区别于动物学 59x） |
| Real_Estate | 房地产 |
| Shopping | 购物 |
| Online_Communities | 在线社区 |
| Internet_and_Telecom | 互联网电信 |
| Home_and_Garden | 家居园艺（区别于家政学 64x） |
| Games | 游戏（区别于博弈论 519.3） |
| Hobbies_and_Leisure | 休闲爱好 |
| Food_and_Drink | 食品饮料（区别于食品科学 664） |
| Sensitive_Subjects | 敏感话题 |

如需覆盖这些类别，需要用 NVIDIA domain-classifier 对 Essential-Web 进行额外标注。

---

## 4. 映射实现方式

### 4.1 基于 FDC Code 前缀

映射基于 FDC code 的前 2 位数字（百位+十位），不依赖 level_2 文本标签：

```python
# 示例
# FDC code "746.92" → 取前缀 "74" → category 17 (Arts_and_Entertainment)
# FDC code "510"    → 取前缀 "51" → category 10 (Mathematics)
# FDC code ""       → category 22 (Other)
# 解析失败          → category 22 (Other)
```

### 4.2 前缀映射表

```python
FDC_PREFIX_TO_DOMAIN = {
    # 0: Computers_and_Electronics
    "00": 0,
    # 1: News_and_General_Works
    "01": 1, "02": 1, "03": 1, "04": 1, "05": 1,
    "06": 1, "07": 1, "08": 1, "09": 1,
    # 2: Philosophy_and_Psychology
    "10": 2, "11": 2, "12": 2, "13": 2, "14": 2,
    "15": 2, "16": 2, "17": 2, "18": 2, "19": 2,
    # 3: Religion
    "20": 3, "21": 3, "22": 3, "23": 3, "24": 3,
    "25": 3, "26": 3, "27": 3, "28": 3, "29": 3,
    # 4: Law_and_Government
    "32": 4, "34": 4,
    # 5: Economics_and_Finance
    "33": 5,
    # 6: Education
    "37": 6,
    # 7: People_and_Society
    "30": 7, "31": 7, "35": 7, "36": 7, "38": 7, "39": 7, "92": 7,
    # 8: English_Language
    "40": 8, "41": 8, "42": 8,
    # 9: Other_Languages
    "43": 9, "44": 9, "45": 9, "46": 9, "47": 9, "48": 9, "49": 9,
    # 10: Mathematics
    "51": 10,
    # 11: Physics_and_Chemistry
    "53": 11, "54": 11,
    # 12: Earth_and_Life_Sciences
    "50": 12, "52": 12, "55": 12, "56": 12, "57": 12, "58": 12, "59": 12,
    # 13: Medicine_and_Health
    "61": 13,
    # 14: Business_and_Management
    "65": 14,
    # 15: Engineering
    "60": 15, "62": 15, "66": 15, "67": 15, "68": 15, "69": 15,
    # 16: Agriculture
    "63": 16,
    # 17: Arts_and_Entertainment
    "70": 17, "71": 17, "72": 17, "73": 17, "74": 17,
    "75": 17, "76": 17, "77": 17, "78": 17,
    # 18: Sports_and_Recreation
    "79": 18,
    # 19: Books_and_Literature
    "80": 19, "81": 19, "82": 19, "83": 19, "84": 19,
    "85": 19, "86": 19, "87": 19, "88": 19, "89": 19,
    # 20: History
    "90": 20, "93": 20, "94": 20, "95": 20, "96": 20,
    "97": 20, "98": 20, "99": 20,
    # 21: Geography_and_Travel
    "91": 21,
    # 22: Home_Economics
    "64": 22,
}
```

### 4.3 提取函数设计

```python
def extract_domain_level_2(eai_taxonomy) -> int:
    """
    从 eai_taxonomy 提取 FDC code 前缀，映射到 23 类 domain。

    Returns:
        int: domain ID (0-22), 其中 22=Home_Economics; -1 表示丢弃
    """
    # 1. 解析 eai_taxonomy
    # 2. 获取 primary FDC code
    # 3. 取前 2 位字符作为前缀
    # 4. 查表 FDC_PREFIX_TO_DOMAIN
    # 5. 未找到 / L2 缺失 / L1 为空 / 解析失败 → 22 (Other)
```

---

## 5. 与 L1 方案的对比

| 维度 | L1 (当前) | L2 映射 (本方案) |
|------|-----------|-----------------|
| 类别数 | 10 | 23 (全部有效) |
| Science 粒度 | 合1 | 拆4 (Math/Phys/Bio/Med) |
| Business 粒度 | 合1 | 拆3 (Mgmt/Eng/Agri) |
| Social Sciences 粒度 | 合1 | 拆4 (Law/Econ/Edu/People) |
| 实现复杂度 | 低 | 中（需解析 FDC code） |
| 依赖 | level_1 文本标签 | FDC code 前缀 |
| 核心价值 | — | **提高域内同质性**，使 DCLM 质量筛选更精准 |

---

## 6. 风险与注意事项

### 6.1 数据量分布

从 10 类扩展到 24 类后，部分类别数据量可能过小：
- Mathematics (51x)：web 上纯数学内容占比极低
- Physics_and_Chemistry (53x-54x)：可能数据量不足
- Other_Languages (43x-49x)：非英语语言在 web 上分布不均

**建议**：实现后先扫描 Essential-Web 各 L2 的实际数据量分布，确认无空类。

### 6.2 FDC Code 解析

- 部分文档的 FDC code 可能为空或格式异常
- `level_2` 标签为空字符串时，FDC code 可能只有百位（如 `508`），需处理 3 位 code 的情况
- 所有无法精确映射到 L2 的文档直接**丢弃**（domain=-1）
  - L2 缺失但 L1 存在 → 丢弃
  - L1 也为空 → 丢弃
  - 解析失败 → 丢弃
  - **理由**：实测仅 8,462 条（0.02%），且数据质量差（avg tok/doc=14,447）。硬塞进某个 L2 类会引入噪声，保留 Other 类会增加无效搜索维度

### 6.3 向后兼容

- 新方案与现有 L1 方案并存，通过 `--domain-level` 参数切换
- 现有 `DOMAIN_MAP` 和 `DOMAIN_NAMES` 保留，新增 `FDC_PREFIX_TO_DOMAIN_L2` 和 `DOMAIN_NAMES_L2`
- `QuaDMixConfig.num_domains` 根据选择的 level 自动适配（10 或 23）

---

## 7. 后续步骤

1. **扫描 Essential-Web 数据**：统计各 FDC 前缀的实际文档数和 token 数，确认 22 个有效类无空类
2. **实现映射代码**：在 `constants.py` 新增 `FDC_PREFIX_TO_DOMAIN_L2`、`DOMAIN_NAMES_L2`，在 `preprocess_essential_web_v1_sharded.py` 新增 `extract_domain_level_2`
3. **验证映射正确性**：抽样检查映射结果，确认 Other 类占比和组成
4. **重新运行 proxy 实验**：使用 23 类 domain 配置，对比 R² 和方向准确率
5. **对比中训效果**：L1-10 vs L2-23 的 CORE metric 对比
