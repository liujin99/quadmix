# FDC Domain Mapping 设计：从 10 类到 22 类

> 2026-06-30 | 基于 Essential-Web v1.0 FDC 分类体系的领域映射方案（L2 实现完成）

## 1. 背景

### 1.1 原 L1 方案

QuaDMix 最初使用 Essential-Web v1.0 的 FDC (Free Decimal Correspondence) L1 分类，共 **10 个领域**（Dewey Decimal 体系）：

| ID | L1 Domain | FDC Code | 占比 |
|----|-----------|----------|------|
| 0 | Industrial arts, Technology, and Engineering | 6xx | 41.1% |
| 1 | Social sciences | 3xx | 22.0% |
| 2 | Science and Natural history | 5xx | 18.4% |
| 3 | Religion | 2xx | 1.8% |
| 4 | Philology; or, Language and languages | 4xx | 0.5% |
| 5 | Literature | 8xx | 1.3% |
| 6 | History and Geography | 9xx | 2.5% |
| 7 | General works, books and libraries, information sciences | 0xx | 8.7% |
| 8 | Philosophy and psychology | 1xx | 1.0% |
| 9 | Arts | 7xx | 2.7% |

**问题**：10 类粒度太粗。例如 "Industrial" (6xx) 占 41.1%，内部混杂工程、医学、商业、农业等异质内容，DCLM 质量分在异质域内区分度低。

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

将 ~100 个 FDC L2 映射到 **22 个领域**，对齐业界经验证的分类粒度。

**设计动机**：L2 映射的核心价值不是改变域比例，而是**提高域内同质性**。当前 FDC L1 的 10 域太粗（如 Industrial 41% 内部混杂工程、医学、商业），DCLM 质量分在异质域内区分度低。拆分后每个 L2 子域更同质，质量筛选更精准。

### 1.4 实现状态

**已完成**（2026-06-30）：
- 映射代码实现：`src/quadmix/constants.py` 中的 `FDC_PREFIX_TO_DOMAIN`
- 预处理脚本更新：`scripts/preprocess/preprocess_essential_web_v1_sharded.py`
- 直接替换 L1 方案（不做双模式兼容）
- 500 shards 数据预处理完成，22 域分布验证通过

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

## 3. FDC-22 映射方案

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
| 9 | **Mathematics** | 51x | 1 | Science(部分) | C17 |
| 10 | **Physics_and_Chemistry** | 53x-54x | 2 | Science(部分) | C6, C7 |
| 11 | **Earth_and_Life_Sciences** | 50x, 52x, 55x-59x | 7 | Science(部分) | C8, C9 |
| 12 | **Medicine_and_Health** | 61x | 1 | Health | C1, C4, C10 |
| 13 | **Business_and_Management** | 65x | 1 | Business_and_Industrial(部分) | — |
| 14 | **Engineering** | 60x, 62x, 66x-69x | 6 | Business_and_Industrial(部分) | C15 |
| 15 | **Agriculture** | 63x | 1 | Food_and_Drink(部分) | C3, C5(部分) |
| 16 | **Arts_and_Entertainment** | 70x-78x | 9 | Arts_and_Entertainment | C13, C21 |
| 17 | **Sports_and_Recreation** | 79x | 1 | Sports | — |
| 18 | **Books_and_Literature** | 8xx | 10 | Books_and_Literature | C13(部分) |
| 19 | **History** | 90x, 93x-99x | 8 | People_and_Society(部分) | C14, C19 |
| 20 | **Geography_and_Travel** | 91x | 1 | Travel_and_Transportation | C14(部分) |
| 21 | **Home_Economics** | 64x | 1 | Home_and_Garden(部分) | — |

**总计：22 类**（全部有效领域）

> **丢弃决策**：
> - Other 类（FDC code=-1，8,462 docs，0.02%）：数据质量差（avg tok/doc=14,447）
> - Other_Languages（43x-49x，64K docs，0.16%）：非英语内容，benchmark 全英语，无贡献

### 3.3 关键决策说明

| 决策 | 理由 |
|------|------|
| Science 拆 4 类 (9-12) | NVIDIA 合1 太粗；ClimbMix 拆成 C6-C9 四个 cluster，验证了拆分的合理性 |
| Philosophy + Psychology 合1 (ID 2) | Web 上哲学数据极少，单独成类会近乎空；NVIDIA 无对应独立类 |
| Religion 独立 (ID 3) | 与 Philosophy 分离，web 上宗教数据量尚可 |
| People_and_Society 吸收 Biography (92x) | 对齐 NVIDIA 的 People_and_Society 范畴 |
| Business 拆 Management (65x) + Engineering (60x/62x/66-69x) | NVIDIA 合为 Business_and_Industrial 太粗，工程 vs 管理词汇差异大 |
| Arts 合 9 个 L2 (ID 16) | 对齐 NVIDIA 的 Arts_and_Entertainment，web 上艺术类同质性高 |
| English vs Other Languages 分离 (8, 丢弃) | Web 数据以英语为主，非英语语言对 multilingual 能力有独立贡献 |
| History 从 Geography 分离 (19, 20) | 对齐 NVIDIA 的 Travel_and_Transportation 独立 |
| **Other 类直接丢弃** | FDC code=-1 的文档仅 8,462 条（0.02%），且 avg tok/doc=14,447（异常高），数据质量差 |
| **Other_Languages 直接丢弃** | 43x-49x 非英语语言仅 64K 条（0.16%），benchmark 全英语无贡献，减少 9 个无效搜索参数 |
| **Home_Economics 独立成域 (ID 21)** | 64x（家政学/消费者科学）占 11.6%，与 People_and_Society 内容风格差异大（食谱 vs 社会学）。独立成域提高同质性，使 DCLM 质量分更有区分度 |
| **60x/66x-69x 归入 Engineering (ID 14)** | 化工、制造、建筑等工业技术与工程设计语义一致，合并后 Engineering 从 8.4% 升至 13.4%，可接受 |

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
# FDC code "746.92" → 取前缀 "74" → category 16 (Arts_and_Entertainment)
# FDC code "510"    → 取前缀 "51" → category 9 (Mathematics)
# FDC code "640"    → 取前缀 "64" → category 21 (Home_Economics)
# FDC code "45"     → 取前缀 "45" → -1 (Other_Languages, 丢弃)
# 解析失败          → -1 (丢弃)
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
    # 9: Mathematics
    "51": 9,
    # 10: Physics_and_Chemistry
    "53": 10, "54": 10,
    # 11: Earth_and_Life_Sciences
    "50": 11, "52": 11, "55": 11, "56": 11, "57": 11, "58": 11, "59": 11,
    # 12: Medicine_and_Health
    "61": 12,
    # 13: Business_and_Management
    "65": 13,
    # 14: Engineering
    "60": 14, "62": 14, "66": 14, "67": 14, "68": 14, "69": 14,
    # 15: Agriculture
    "63": 15,
    # 16: Arts_and_Entertainment
    "70": 16, "71": 16, "72": 16, "73": 16, "74": 16,
    "75": 16, "76": 16, "77": 16, "78": 16,
    # 17: Sports_and_Recreation
    "79": 17,
    # 18: Books_and_Literature
    "80": 18, "81": 18, "82": 18, "83": 18, "84": 18,
    "85": 18, "86": 18, "87": 18, "88": 18, "89": 18,
    # 19: History
    "90": 19, "93": 19, "94": 19, "95": 19, "96": 19,
    "97": 19, "98": 19, "99": 19,
    # 20: Geography_and_Travel
    "91": 20,
    # 21: Home_Economics
    "64": 21,
}
```

### 4.3 提取函数设计

```python
def extract_domain_level_2(eai_taxonomy) -> int:
    """
    从 eai_taxonomy 提取 FDC code 前缀，映射到 22 类 domain。

    Returns:
        int: domain ID (0-21), 其中 21=Home_Economics; -1 表示丢弃
    """
    # 1. 解析 eai_taxonomy (JSON string → dict)
    # 2. 获取 primary FDC code
    # 3. 取前 2 位字符作为前缀
    # 4. 查表 FDC_PREFIX_TO_DOMAIN
    # 5. 未找到 / 解析失败 → -1 (丢弃)
```

**丢弃规则**：
- FDC code=-1 或解析失败 → 丢弃（8,462 条，0.02%）
- 43x-49x (Other_Languages) → 丢弃（64K 条，0.16%）
- 未映射前缀 → 丢弃（实际无，所有有效前缀已覆盖）

---

## 5. 与 L1 方案的对比

| 维度 | L1 (原方案) | L2 映射 (本方案) |
|------|-----------|-----------------|
| 类别数 | 10 | 22 (全部有效) |
| Science 粒度 | 合1 | 拆4 (Math/Phys/Bio/Med) |
| Business 粒度 | 合1 | 拆3 (Mgmt/Eng/Agri) |
| Social Sciences 粒度 | 合1 | 拆4 (Law/Econ/Edu/People) |
| Industrial 粒度 | 合1 (41.1%) | 拆4 (Eng 13.4%, Business 7.2%, Medicine 6.5%, Agriculture 2.4%) |
| 实现复杂度 | 低 | 中（需解析 FDC code） |
| 依赖 | level_1 文本标签 | FDC code 前缀 |
| 核心价值 | — | **提高域内同质性**，使 DCLM 质量筛选更精准 |

---

## 6. L2 分布实测结果

基于 500 shards 预处理数据（2026-06-30 完成）：

### 6.1 总体统计

| 指标 | 数值 |
|------|------|
| 总文档数 | 40,343,674 |
| 总 token 数（估计） | 38,489,683,203 |
| 丢弃文档数 | 72,190 (0.18%) |
| 有效域数 | 22 |
| Max/Min 比例 | 50.5x |
| Top-5 域占比 | 52.5% |
| Top-10 域占比 | 84.3% |
| Bottom-5 域占比 | 2.91% |

### 6.2 各域分布

| ID | Domain | Docs | % | Tokens (est) | % | Avg tok/doc |
|----|--------|------|---|--------------|---|-------------|
| 14 | Engineering | 5,406,635 | 13.40% | 4,302,206,226 | 11.18% | 796 |
| 21 | Home_Economics | 4,702,323 | 11.66% | 2,720,186,097 | 7.07% | 578 |
| 17 | Sports_and_Recreation | 4,209,970 | 10.44% | 3,561,650,280 | 9.25% | 846 |
| 7 | People_and_Society | 3,635,406 | 9.01% | 3,480,102,840 | 9.04% | 957 |
| 16 | Arts_and_Entertainment | 3,228,105 | 8.00% | 2,085,796,757 | 5.42% | 646 |
| 0 | Computers_and_Electronics | 3,032,580 | 7.52% | 3,792,599,502 | 9.85% | 1251 |
| 13 | Business_and_Management | 2,908,173 | 7.21% | 3,192,910,562 | 8.30% | 1098 |
| 5 | Economics_and_Finance | 2,656,396 | 6.58% | 3,095,893,993 | 8.04% | 1165 |
| 12 | Medicine_and_Health | 2,637,260 | 6.54% | 3,097,592,638 | 8.05% | 1175 |
| 4 | Law_and_Government | 1,609,379 | 3.99% | 2,186,360,604 | 5.68% | 1359 |
| 6 | Education | 1,095,972 | 2.72% | 1,067,907,316 | 2.77% | 974 |
| 15 | Agriculture | 969,577 | 2.40% | 890,467,942 | 2.31% | 918 |
| 3 | Religion | 741,405 | 1.84% | 837,913,532 | 2.18% | 1130 |
| 20 | Geography_and_Travel | 694,088 | 1.72% | 880,583,022 | 2.29% | 1269 |
| 11 | Earth_and_Life_Sciences | 649,420 | 1.61% | 848,175,111 | 2.20% | 1306 |
| 18 | Books_and_Literature | 510,058 | 1.26% | 637,912,890 | 1.66% | 1251 |
| 1 | News_and_General_Works | 481,019 | 1.19% | 357,534,481 | 0.93% | 743 |
| 2 | Philosophy_and_Psychology | 416,673 | 1.03% | 549,875,245 | 1.43% | 1320 |
| 19 | History | 328,646 | 0.81% | 427,350,136 | 1.11% | 1300 |
| 10 | Physics_and_Chemistry | 179,220 | 0.44% | 197,004,411 | 0.51% | 1099 |
| 8 | English_Language | 144,223 | 0.36% | 141,481,558 | 0.37% | 981 |
| 9 | Mathematics | 107,146 | 0.27% | 138,178,060 | 0.36% | 1290 |

### 6.3 分布特征分析

**大域主导**（>5%）：9 个域占 80.2%
- Engineering (13.4%), Home_Economics (11.7%), Sports (10.4%), People (9.0%), Arts (8.0%), Computers (7.5%), Business (7.2%), Economics (6.6%), Medicine (6.5%)

**中域**（1-5%）：10 个域占 16.9%
- Law (4.0%), Education (2.7%), Agriculture (2.4%), Religion (1.8%), Geography (1.7%), Earth_and_Life (1.6%), Books (1.3%), News (1.2%), Philosophy (1.0%), History (0.8%)

**微域**（<1%）：3 个域占 1.1%
- Physics (0.4%), English (0.4%), Mathematics (0.3%)

**与 L1 对比**：
- L1 Industrial (41.1%) → L2 Engineering(13.4) + Business(7.2) + Medicine(6.5) + Agriculture(2.4) = 29.5%
- L1 Social (22.0%) → L2 People(9.0) + Economics(6.6) + Law(4.0) + Education(2.7) = 22.3%（吻合）
- L1 Arts (2.7%) → L2 Arts(8.0) + Sports(10.4) = 18.4%（原 L1 Arts 实际是 7xx，包含 Sports）
- **中间层更均匀**：L2 有 9 个域在 6.5%-10.4%，L1 只有 1 个域 <20%
- **Max/Min 改善**：50.5x vs 82x（L1），因为去除了 Other_Languages 和无效数据

### 6.4 丢弃数据统计

| 类别 | FDC 前缀 | 文档数 | 占比 | 原因 |
|------|----------|--------|------|------|
| Other_Languages | 43x-49x | 64,000 | 0.16% | 非英语，benchmark 全英语无贡献 |
| Invalid FDC | -1 | 8,462 | 0.02% | 数据质量差（avg tok/doc=14,447） |
| **总计** | — | 72,190 | 0.18% | — |

**丢弃前缀分布**（Top-5）：
- 49 (Other_Languages): 39,869 docs (55.2%)
- 46 (Other_Languages): 8,439 docs (11.7%)
- -1 (Invalid): 8,423 docs (11.7%)
- 44 (Other_Languages): 6,675 docs (9.2%)
- 43 (Other_Languages): 4,315 docs (6.0%)

---

## 7. 分析与设计过程

### 7.1 L1 域配比约束分析

通过 874 组 proxy 实验分析发现，当前采样机制下域配比变化受限：

**结构性约束**：
- 采样公式：S(r̄) = (2/(1+e^{-λ(ω-r̄)}))^η + ε → S_max ≤ 2.001（单文档最多重复 2 次）
- ω 硬约束：ω ∈ [0, 0.1]，只有前 10% 质量文档参与采样
- 实测域采样率：2.42%-6.48%，最大/最小比仅 2.7x

**PCA 分析**：
- PC1+PC2+PC3 解释 97.4% 方差
- 本质上只在 3 个大域之间做 tradeoff
- 7 个小域始终边缘化（每个 <5%）
- **有效搜索空间是 3 维而非 10 维**

**结论**：域配比约束是核心问题，不是加实验数或换回归模型能解决的。

### 7.2 L2 映射策略转变

**初期思路**：通过 L2 映射改变域比例
- 问题：L1 的 10 域太粗，搜索空间退化
- 期望：L2 的 22 域能扩大有效搜索空间

**转变后的思路**：通过 L2 映射提高域内同质性
- 核心价值：让 DCLM 质量分在更同质的域内更有区分度
- 例：L1 Industrial (41%) 内部混杂工程、医学、商业，DCLM 高分文档可能集中在某个子主题
- L2 拆分后，Engineering (13.4%)、Medicine (6.5%)、Business (7.2%) 各自更同质

### 7.3 Fallback 策略讨论

**问题**：L2 缺失率 11.2%（4.5M docs），其中 General works 占 86.3%

**方案对比**：
- 方案 A：每 L1 一个 Other → 34 类（用户认为太多）
- 方案 B：单一 Other 类 → 23 类（22 有效 + 1 Other）
- **最终方案**：直接丢弃未映射数据 → 22 类（全部有效）

**决策理由**：
- Other 类（FDC code=-1）仅 8,462 条（0.02%），数据质量差
- Other_Languages（43x-49x）仅 64K 条（0.16%），非英语无贡献
- 丢弃后所有 22 域都是有效域，无需处理 Other 类的同质性问题

### 7.4 域权重参数讨论

**问题**：大域主导（9 个域占 80.2%），搜索本质上只在这 9 域间做 ±2-3% 微调

**方案**：引入显式域权重参数 `domain_weight[m]`，让搜索直接控制域比例

**用户决策**：先不引入，用当前 22 域方案实验验证 L2 同质性改善是否足够
- 理由：增加参数会扩大搜索空间，需要更多实验
- 优先级：先验证 L2 映射本身的价值

---

## 8. 风险与注意事项

### 8.1 微域冗余问题

7 个微域（<1%）占 2.7%，但增加 63 个无效搜索参数（29%）：
- Physics (0.4%), English (0.4%), Mathematics (0.3%)
- History (0.8%), Books (1.3%), News (1.2%), Philosophy (1.0%)

**当前策略**：保留微域
- 理由：Physics/Math/English 对科学推理和语言理解有独立贡献
- 风险：搜索效率降低，但可接受

**备选方案**（待实验验证后决定）：
- 合并微域到相近的大域（如 Physics → Earth_and_Life_Sciences）
- 引入域权重参数，让搜索自动调整

### 8.2 FDC Code 解析

- 部分文档的 FDC code 可能为空或格式异常
- `level_2` 标签为空字符串时，FDC code 可能只有百位（如 `508`），需处理 3 位 code 的情况
- 已实现：解析失败 → -1（丢弃），实测仅 39 条（0.0001%）

### 8.3 实现方式

**直接替换 L1**（用户选择方案 B）：
- 不做双模式兼容（无 `--domain-level` 参数）
- 预处理输出直接覆盖 L1 数据
- `constants.py` 中 `DOMAIN_NAMES` 和 `FDC_PREFIX_TO_DOMAIN` 已是 L2 版本
- `NUM_DOMAINS = 22`

**兼容性处理**：
- `report.py` 动态检测域数量（`_get_domain_short(num_domains)`）
- 支持 L1 (10 域) 和 L2 (22 域) 的历史数据可视化

---

## 9. 后续步骤

### 9.1 当前状态（2026-06-30）

- [x] FDC 映射设计完成
- [x] 映射代码实现（`constants.py`, `preprocess_essential_web_v1_sharded.py`）
- [x] 500 shards 数据预处理完成
- [x] L2 分布验证通过（22 域，无空类）
- [ ] L2 proxy 实验（500 次）
- [ ] 对比 L1 vs L2 的 R² 和方向准确率
- [ ] 中训验证（L2 配比的 CORE metric）

### 9.2 下一步

**L2 proxy 实验**：
1. 运行 500 次 proxy 实验（使用 22 域预处理数据）
2. 训练 21 个 per-task LightGBM 模型
3. 对比 R² 和方向准确率是否提升
4. 重点关注：winogrande, piqa, commonsense_qa 等高 R² 但方向错的任务

**中训验证**：
1. 使用 L2 proxy 搜索结果生成训练数据
2. 运行中训实验（842M tokens，与 r2_sigma_weighted 同配置）
3. 对比 CORE metric（目标：>0.2746）

**可选优化**（待实验验证后决定）：
- 引入域权重参数（如果 L2 同质性改善不足）
- 合并微域（如果搜索效率严重下降）
- 放宽 ω 约束（从 [0, 0.1] 扩到 [0, 0.3]）
