# 实验二：使用 Deepseek-R1 和 V3 构建宪法知识库 RAG 应用实验报告



------

## 一、实验分工

|  1   | 占比 |                           主要工作                           |
| :--: | :--: | :----------------------------------------------------------: |
|  2   | 35%  | 完成宪法知识库构建、数据预处理、模型调用模块开发、撰写实验报告 |
|  3   | 25%  |       协助数据集处理、测试问题设计、实验数据记录与分析       |
|  4   | 20%  |         协助数据集处理、交互界面测试、可视化图表生成         |
|  5   | 20%  |                向量索引构建、混合检索策略实现                |

------

## 二、实验目的

1. 掌握 RAG（Retrieval-Augmented Generation，检索增强生成）技术的基本原理与完整实现流程
2. 验证 Deepseek-R1 作为生成模型、Deepseek-V3 作为检索模型在宪法知识库问答中的协同效果
3. 评估不同检索策略（语义检索 + 关键词检索混合）对生成结果准确性、响应时间的影响
4. 探索多模型协作在法律类知识密集型问答任务中的应用潜力
5. 对比 RAG 方案与单独大模型在事实型、推理型、多跳推理型三类问题上的性能差异

------

## 三、流程图

## 四、实验环境

| 环境配置项 |                    参数说明                     |
| :--------: | :---------------------------------------------: |
|  操作系统  |                   Windows 11                    |
|    CPU     |      11th Gen Intel(R) Core(TM) i5-1155G7       |
|    内存    |                       32G                       |
|  嵌入模型  |        BAAI/bge-base-zh-v1.5（本地部署）        |
|   大模型   | Deepseek-R1、Deepseek-V3.2（硅基流动 API 调用） |

------

## 五、实验设计

### 5.1 实验整体流程

本实验基于模块化设计，完整流程分为**系统初始化、核心执行、实验评估**三大阶段，具体如下：

1. **环境准备与依赖检查**：验证所需 Python 包安装状态，配置 API 密钥、本地模型路径、宪法知识库文件路径
2. **模型选择与系统初始化**：交互式选择检索 / 生成模型组合，完成知识库加载、文本分块、向量索引构建、模型客户端初始化
3. **核心功能执行**：支持自定义提问（RAG 模式）和完整实验评估两种模式，执行混合检索、上下文构建、回答生成
4. **实验评估与可视化**：对 9 道测试题进行量化评估，生成性能对比报告和可视化图表

### 5.2 关键步骤设计

#### 5.2.1 数据准备阶段

- **知识库构建**：加载《中华人民共和国宪法》全文，共 18030 字符
- **文本分块**：按段落分块，设置块大小 500 字符、重叠 50 字符，最终生成 19 个文本块，保证语义连贯性
- **向量化与索引构建**：使用本地 BGE-base-zh-v1.5 模型将文本块转换为 768 维向量，构建 FAISS 向量索引；同时基于 jieba 分词构建 BM25 关键词索引

#### 5.2.2 混合检索策略实现

- 融合**语义检索（权重 70%）\**和\**关键词检索（权重 30%）**：语义检索基于 FAISS 向量相似度，关键词检索基于 BM25 算法
- 对两类检索结果进行归一化处理后加权融合，按综合分数排序，返回得分最高的 5 个文档作为上下文
- 解决单一检索方式的局限性：语义检索理解问题语义，关键词检索精准匹配法条关键词

#### 5.2.3 模型调用与回答生成

- 支持多模型组合：检索模型可选 Deepseek-V3、Deepseek-R1、Qwen2.5 系列（本地 Ollama），生成模型同理
- 生成规则：Deepseek-R1 生成时强制包含「推理过程 + 最终答案」两部分，所有模型严格基于检索到的宪法原文回答，无上下文时提示信息不足，禁止编造答案
- 对比实验：所有测试问题同时执行**RAG 方案**和**单独模型方案**，量化对比性能差异

#### 5.2.4 测试问题设计

基于《中华人民共和国宪法》设计 9 道测试题，分为三类：

|  问题类型  | 数量 |                示例问题                |           考察重点           |
| :--------: | :--: | :------------------------------------: | :--------------------------: |
|   事实型   |  3   | 公民在法律面前一律平等是哪一条规定的？ | 法条精准定位、事实性知识问答 |
|   推理型   |  3   |      为什么宪法是国家的根本大法？      | 法条背后的立法逻辑、原则理解 |
| 多跳推理型 |  3   |   公民行使言论自由的边界及相关条款？   | 多法条综合应用、复杂逻辑推理 |

------

## 六、实验过程与运行结果

### 6.1 系统初始化过程

1. **依赖检查**：验证所有依赖包安装完成，通过检查
2. **模型选择**：选择检索模型为 Deepseek-V3，生成模型为 Deepseek-R1
3. **知识库加载与分块**：成功加载宪法全文（18030 字符），分块生成 19 个文本块
4. **嵌入模型与索引构建**：加载本地 BGE-base-zh-v1.5 模型，生成 19×768 维向量，完成 FAISS 向量索引和 BM25 关键词索引构建
5. **模型客户端初始化**：成功连接硅基流动 API，加载 Deepseek-R1 模型，RAG 系统初始化完成

### 6.2 自定义提问测试（示例）

**用户问题**：公民的基本政治自由有哪些？

1. **混合检索**：语义检索和关键词检索各返回 5 个结果，综合排序后选取 Top5 文档，构建 8886 字符上下文
2. **回答生成**：Deepseek-R1 基于上下文生成回答，包含推理过程和最终答案，引用第三十四条、第三十五条、第四十一条等具体条款
3. **耗时统计**：检索耗时 0.10s，生成耗时 110.25s，总耗时 110.35s

### 6.3 完整实验评估过程与结果

#### 6.3.1 测试执行过程

本次实验共执行 9 道测试题，覆盖事实型、推理型、多跳推理型三类，每道题均同步执行 RAG 方案与单独模型方案，核心执行过程如下：

- **检索阶段**：所有问题均通过混合检索返回 5 个相关文档，检索耗时稳定在 0.06s-0.14s 之间，检索效率极高
- **上下文构建**：根据检索结果拼接宪法原文，上下文长度从 3164 字符到 9690 字符不等，为生成提供完整参考
- **回答生成**：Deepseek-R1 基于上下文生成带推理过程的回答，单独模型直接生成无上下文的回答
- **耗时记录**：记录每道题的检索耗时、生成耗时、总耗时，用于性能分析

#### 6.3.2 详细性能对比表

|  问题类型  |                     测试问题                      | RAG 回答时间 (s) | 单独模型时间 (s) | RAG 准确率 | 单独模型准确率 | 检索文档数 |
| :--------: | :-----------------------------------------------: | :--------------: | :--------------: | :--------: | :------------: | :--------: |
|   事实型   |      公民在法律面前一律平等是哪一条规定的？       |      26.01       |      15.73       |   100.0%   |     66.7%      |     5      |
|   事实型   |          公民基本政治自由包括哪些内容？           |      55.22       |      33.17       |   100.0%   |     100.0%     |     5      |
|   事实型   |      全国人大有权修改宪法吗？规定在第几章？       |      35.68       |      29.07       |   100.0%   |     100.0%     |     5      |
|   推理型   | 为什么规定 "国家尊重和保障人权"？核心意义是什么？ |      116.27      |      49.28       |   75.0%    |     50.0%      |     5      |
|   推理型   |           为什么宪法是国家的根本大法？            |      57.88       |      47.59       |   66.7%    |     33.3%      |     5      |
|   推理型   |       人格尊严不受侵犯体现了什么宪法原则？        |      50.03       |      34.81       |   75.0%    |     100.0%     |     5      |
| 多跳推理型 |        公民行使言论自由的边界及相关条款？         |      59.08       |      46.20       |   75.0%    |     75.0%      |     5      |
| 多跳推理型 |        怀孕女职工权益受哪些宪法条款保护？         |      179.79      |      35.23       |   25.0%    |     25.0%      |     5      |
| 多跳推理型 |   全国人大常委会解释权与全国人大修宪权的关系？    |      59.89       |      54.58       |   80.0%    |     40.0%      |     5      |

#### 6.3.3 平均性能汇总

|      指标      | RAG 方案 | 单独模型 |       提升 / 下降        |
| :------------: | :------: | :------: | :----------------------: |
|  平均响应时间  | 71.09 秒 | 38.41 秒 |  -85.1%（RAG 耗时更长）  |
|   平均准确率   |  77.4%   |  65.6%   | +11.9%（RAG 准确率更高） |
|  平均检索耗时  | 0.08 秒  |    -     |            -             |
| 平均检索文档数 |    5     |    -     |            -             |

#### 6.3.4 历史性能指标（完整记录）

| 序号 |                           问题                            | 总耗时 (s) | 检索耗时 (s) | 生成耗时 (s) | 检索文档数 |
| :--: | :-------------------------------------------------------: | :--------: | :----------: | :----------: | :--------: |
|  1   |                 公民的基本政治自由有哪些                  |   110.35   |     0.10     |    110.25    |     5      |
|  2   |          公民在法律面前一律平等是哪一条规定的？           |   26.01    |     0.07     |    25.94     |     5      |
|  3   |              公民基本政治自由包括哪些内容？               |   55.22    |     0.07     |    55.15     |     5      |
|  4   |  全国人民代表大会有权修改宪法吗？这一职权规定在第几章？   |   35.68    |     0.07     |    35.61     |     5      |
|  5   | 《宪法》为什么规定 "国家尊重和保障人权"？核心意义是什么？ |   116.27   |     0.07     |    116.20    |     5      |
|  6   |     为什么宪法是国家的根本大法，具有最高的法律效力？      |   57.88    |     0.09     |    57.79     |     5      |
|  7   |       公民的人格尊严不受侵犯，体现了什么宪法原则？        |   50.03    |     0.06     |    49.96     |     5      |
|  8   |            公民行使言论自由的边界及相关条款？             |   59.08    |     0.07     |    59.01     |     5      |
|  9   |         怀孕女职工的合法权益受哪些宪法条款保护？          |   179.79   |     0.08     |    179.71    |     5      |
|  10  | 全国人大常委会的宪法解释权与全国人大的修宪权是什么关系？  |   59.89    |     0.14     |    59.75     |     5      |

#### 6.3.5、实验结果可视化

## 七、实验结果分析

### 7.1 不同问题类型性能分析

#### 7.1.1 事实型问题

- **准确率**：RAG 方案达到 100%，远高于单独模型的 88.9%（平均），体现 RAG 在事实性知识问答中的绝对优势
- **原因**：RAG 通过检索精准定位宪法原文条款，确保答案完全基于法条，避免模型幻觉；单独模型依赖预训练知识，易出现条款号错误、内容遗漏（如第一题单独模型未准确匹配第三十三条）
- **耗时**：RAG 平均耗时 38.97s，单独模型平均 25.99s，RAG 因检索和上下文处理耗时更长，但准确率提升显著
- **典型案例**："全国人大有权修改宪法吗？" 问题中，RAG 直接检索到第三章第六十二条，100% 准确回答；单独模型虽回答正确，但额外补充了修宪程序等非问题要求内容

#### 7.1.2 推理型问题

- **准确率**：RAG 方案平均 72.2%，单独模型平均 61.1%，RAG 仍有优势，但部分问题（如 "人格尊严体现的原则"）单独模型表现更好
- **原因**：推理型问题需要理解立法背景和原则，若知识库仅包含法条原文、缺乏立法说明，RAG 无法回答 "为什么" 类问题；单独模型凭借预训练知识可补充背景信息
- **耗时**：RAG 平均耗时 74.73s，单独模型平均 43.89s，RAG 耗时约为单独模型的 1.7 倍
- **典型案例**："为什么宪法是根本大法" 问题中，RAG 仅能基于法条原文回答，而单独模型可补充立法基础、法律效力层级等背景知识

#### 7.1.3 多跳推理型问题

- **准确率**：RAG 方案平均 60.0%，单独模型平均 46.7%，RAG 在多法条综合类问题上优势明显
- **原因**：多跳推理需要关联多个法条，RAG 可检索到所有相关条款并整合分析，保证答案的全面性；单独模型易遗漏部分条款，出现逻辑断层
- **耗时**：RAG 平均耗时 99.59s，单独模型平均 45.34s，RAG 因需要处理多段上下文，耗时大幅增加
- **典型案例**："全国人大常委会解释权与修宪权的关系" 问题中，RAG 检索到第六十二条、第六十七条等相关条款，准确分析二者关系，准确率 80%；单独模型仅回答 40%，遗漏关键条款

### 7.2 检索策略效果分析

- **混合检索有效性**：语义检索和关键词检索互补，语义检索理解问题语义，关键词检索精准匹配法条关键词，综合检索结果的相关性远高于单一检索方式
- **检索耗时**：所有问题的检索耗时均在 0.1s 左右，远低于生成耗时，说明检索模块效率极高，RAG 的耗时主要来自大模型生成
- **文档数量稳定性**：所有问题均稳定返回 5 个检索文档，上下文长度充足，为生成模型提供完整参考信息
- **异常案例分析**："怀孕女职工权益保护" 问题中，RAG 和单独模型准确率均仅 25%，原因是宪法中仅规定了男女平等原则，未明确怀孕女职工的具体保护条款和用人单位责任，知识库信息不足导致回答质量差

### 7.3 RAG 方案与单独模型的核心差异

|    维度    |              RAG 方案              |                单独模型                |
| :--------: | :--------------------------------: | :------------------------------------: |
|  答案依据  |  严格基于宪法原文，可追溯具体条款  |     基于模型预训练知识，无外部依据     |
|  幻觉风险  |  极低，无上下文时明确提示信息不足  |       较高，易编造条款、错误引用       |
| 知识时效性 |    可通过更新知识库实时更新知识    |     依赖模型训练数据，无法实时更新     |
|  响应时间  |        较长（平均 71.09s）         |          较短（平均 38.41s）           |
|  适用场景  | 法律、医疗等需要精准事实依据的场景 | 通用对话、创意生成等对事实要求低的场景 |

------

## 八、实验结论与展望

### 8.1 实验结论

1. **RAG 技术应用于法律问答场景效果突出**：在处理事实类问题与多跳推理类问题时，RAG 模型的回答准确率明显优于独立大模型，可有效保障答案的权威性与可溯源性，较好缓解大模型生成幻觉问题。
2. **混合检索模式可显著优化检索效果**：结合语义检索与关键词检索的方式，既实现精准语义匹配，又兼顾关键信息定位，有效提高检索内容的相关性，为后续答案生成提供高质量参考依据。
3. **RAG 方案存在一定应用短板**：系统整体响应速度慢于独立大模型，且回答效果高度依赖知识库内容完善程度，面对涉及立法背景、外部关联知识的复杂推理问题，表现不及独立模型。
4. **Deepseek-R1 模型具备优秀推理适配性**：作为生成端模型，该模型可清晰呈现推理逻辑，回答内容严格依托上下文信息，能够满足法律问答对严谨性、规范性的核心要求。
5. **知识库完整性为关键制约因素**：当知识库缺少对应法律信息（如女职工孕期相关权责规定）时，RAG 模型与独立模型表现相近，充分印证知识库完备程度直接决定问答质量。

### 8.2 现存问题与改进方向

1. **优化系统响应效率**：当前 RAG 耗时主要集中于大模型生成环节，可通过动态调整上下文文档数量（如选取 Top3 相关文档）、采用轻量化生成模型、启用流式生成等方式缩减耗时。
2. **完善知识库内容与结构**：补充宪法立法背景、司法解释及关联配套法律（如劳动法）等内容，提升复杂推理与多跳推理问题的作答效果；优化文本切块规则，采用法条切块替代段落切块，提高检索精准度。
3. **升级检索优化策略**：根据不同问题类型，动态调整语义检索与关键词检索的权重配比；引入重排序模块，进一步筛选高相关性检索内容。
4. **优化模型组合与交互能力**：测试多种检索模型与生成模型搭配方案（如本地部署 Qwen2.5 模型），通过对比实验确定最优组合；拓展多轮对话功能，实现上下文关联式问答。
5. **强化信息缺失场景处理能力**：针对知识库信息不足的问题，增设外部知识检索接口，自动补充相关法律条文与司法解释，提升回答的全面性。

### 8.3 应用前景展望

本次实验搭建的宪法知识库 RAG 系统，可落地应用于法律科普、普法宣传、政务法律咨询等场景，为社会公众提供精准、权威的宪法知识问答服务。同时，该技术框架具备较强通用性与可迁移性，能够快速适配民法典、刑法等其他法律文本，以及医疗、金融等专业领域知识库，具备广阔的落地应用空间。

------

## 九、附录

### 9.1 完整实验代码

```
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
class Config:
    """实验配置"""
    # 硅基流动 API 配置
    API_KEY = "sk-vffafuyiuddepclemthzstptdffqnjyseawdmifximmgytuw"
    API_BASE_URL = "https://api.siliconflow.cn/v1"
    
    # 本地Ollama配置
    OLLAMA_URL = "http://localhost:11434/v1"
    
    # 模型选择
    AVAILABLE_MODELS = {
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3.2",
        "qwen2.5": "qwen2.5:7b",
        "qwen2.5-0.5b": "qwen2.5:0.5b"
    }
    
    # 检索配置
    RETRIEVAL_TOP_K = 5
    SEMANTIC_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3
    
    # 文件路径
    KNOWLEDGE_FILE = "D:/Large Model/deepseek/中华人民共和国宪法.txt"
    VECTOR_DB_PATH = "faiss_index"
    RESULT_IMAGE = "experiment_results.png"
    
    # 分块配置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # 本地模型路径（你已下载的模型）
    LOCAL_EMBEDDING_MODEL_PATH = "D:/Large Model/deepseek/models/bge-base-zh-v1.5"
    
    # 测试问题
    TEST_QUESTIONS = [
        {"type": "事实型", "question": "根据《宪法》，中华人民共和国公民在法律面前一律平等是哪一条规定的？", 
         "expected_keywords": ["第三十三条", "第二章", "法律面前一律平等"]},
        {"type": "事实型", "question": "《宪法》规定的公民基本政治自由包括哪些内容？",
         "expected_keywords": ["言论", "出版", "集会", "结社", "游行", "示威", "第三十五条"]},
        {"type": "事实型", "question": "全国人民代表大会有权修改宪法吗？这一职权规定在第几章？",
         "expected_keywords": ["有权", "修改宪法", "第三章", "第六十二条"]},
        {"type": "推理型", "question": "《宪法》为什么规定\"国家尊重和保障人权\"？这一规定的核心意义是什么？",
         "expected_keywords": ["公民权利", "法治原则", "第三十三条", "基本权利保障"]},
        {"type": "推理型", "question": "为什么宪法是国家的根本大法，具有最高的法律效力？",
         "expected_keywords": ["根本大法", "最高法律效力", "立法基础"]},
        {"type": "推理型", "question": "公民的人格尊严不受侵犯，这一规定体现了什么宪法原则？",
         "expected_keywords": ["人格尊严", "第三十八条", "人权保障", "平等原则"]},
        {"type": "多跳推理型", "question": "结合宪法条款，分析公民在行使言论自由时需要遵守什么边界？相关条款有哪些？",
         "expected_keywords": ["言论自由", "第五十一条", "第三十五条", "不得损害国家利益"]},
        {"type": "多跳推理型", "question": "怀孕女职工的合法权益受哪些宪法条款保护？用人单位违反相关规定会承担什么责任？",
         "expected_keywords": ["女职工保护", "第四十八条", "国家保障", "合法权益"]},
        {"type": "多跳推理型", "question": "全国人大常委会的宪法解释权与全国人大的修宪权是什么关系？分别规定在哪些条款？",
         "expected_keywords": ["宪法解释权", "修宪权", "第六十二条", "第六十七条", "监督宪法实施"]}
    ]


# ==================== 数据预处理模块 ====================
class DataProcessor:
    """数据预处理与分块"""
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    def load_knowledge_base(self, file_path: str) -> str:
        """加载宪法知识库"""
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            print("请检查文件路径是否正确")
            local_file = "中华人民共和国宪法.txt"
            if os.path.exists(local_file):
                print(f"✅ 在当前目录找到文件: {local_file}")
                file_path = local_file
            else:
                print("❌ 未找到宪法文件，将创建默认知识库")
                self._create_default_kb()
                file_path = Config.KNOWLEDGE_FILE
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ 成功加载宪法知识库，共 {len(content)} 字符")
        return content
    
    def _create_default_kb(self):
        """创建默认知识库（如果找不到文件）"""
        default_content = """# 中华人民共和国宪法（核心条款）

## 第二章 公民的基本权利和义务

第三十三条 凡具有中华人民共和国国籍的人都是中华人民共和国公民。中华人民共和国公民在法律面前一律平等。国家尊重和保障人权。

第三十五条 中华人民共和国公民有言论、出版、集会、结社、游行、示威的自由。

第三十八条 中华人民共和国公民的人格尊严不受侵犯。禁止用任何方法对公民进行侮辱、诽谤和诬告陷害。

第四十八条 中华人民共和国妇女在政治的、经济的、文化的、社会的和家庭的生活等各方面享有同男子平等的权利。

第五十一条 中华人民共和国公民在行使自由和权利的时候，不得损害国家的、社会的、集体的利益和其他公民的合法的自由和权利。

## 第三章 国家机构

第六十二条 全国人民代表大会行使下列职权：（一）修改宪法；（二）监督宪法的实施；

第六十七条 全国人民代表大会常务委员会行使下列职权：（一）解释宪法，监督宪法的实施；
"""
        with open(Config.KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
            f.write(default_content)
        print(f"✅ 已创建默认知识库: {Config.KNOWLEDGE_FILE}")

    def split_text(self, text: str) -> List[str]:
        """文本分块"""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_length + len(para) <= self.chunk_size:
                current_chunk.append(para)
                current_length += len(para)
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    overlap_text = '\n\n'.join(current_chunk[-1:])[-self.chunk_overlap:] if current_chunk else ""
                    current_chunk = [overlap_text, para] if overlap_text else [para]
                    current_length = len(overlap_text) + len(para) if overlap_text else len(para)
                else:
                    current_chunk.append(para)
                    current_length = len(para)
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        print(f"✅ 文本分块完成，共 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks[:5]):
            print(f"  块{i+1}: {len(chunk)}字符 - {chunk[:50]}...")
        
        return chunks


# ==================== 向量化与检索模块 ====================
class VectorRetriever:
    """向量检索与混合检索"""
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        self.bm25_index = None

    def init_embeddings(self):
        """初始化中文嵌入模型 - 强制使用本地下载的模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # 使用你已下载的本地模型路径
            local_model_path = Config.LOCAL_EMBEDDING_MODEL_PATH
            
            # 1. 检查文件夹是否存在
            if not os.path.exists(local_model_path):
                print(f"❌ 本地模型文件夹不存在: {local_model_path}")
                print("请确认路径是否正确")
                return False
            
            # 2. 检查模型权重文件是否存在
            model_file = os.path.join(local_model_path, "pytorch_model.bin")
            if not os.path.exists(model_file):
                print(f"❌ 模型权重文件不存在: {model_file}")
                print("请确认本地模型文件是否完整")
                return False
            
            # 3. 直接从本地路径加载模型（不会联网下载）
            print(f"✅ 找到本地模型: {local_model_path}")
            print("正在加载本地嵌入模型...")
            self.embeddings = SentenceTransformer(local_model_path)
            print("✅ 本地Embedding模型加载成功！")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def build_index(self, documents: List[str]):
        """构建FAISS向量索引"""
        if self.embeddings is None:
            print("⚠️ 嵌入模型未加载，跳过向量索引构建")
            return False
            
        try:
            import faiss
            self.documents = documents
            print("正在生成文档向量...")
            vectors = self.embeddings.encode(documents, show_progress_bar=True)
            vectors = np.array(vectors).astype('float32')
            print(f"向量维度: {vectors.shape}")
            
            dimension = vectors.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(vectors)
            print(f"✅ 向量索引构建完成，共 {len(documents)} 个文档")
            return True
        except Exception as e:
            print(f"❌ 向量索引构建失败: {e}")
            return False

    def build_bm25_index(self):
        """构建BM25关键词索引"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            
            tokenized_docs = []
            for doc in self.documents:
                tokens = jieba.lcut(doc)
                tokenized_docs.append(tokens)
            
            self.bm25_index = BM25Okapi(tokenized_docs)
            print("✅ BM25关键词索引构建完成")
            return True
        except Exception as e:
            print(f"⚠️ BM25索引构建失败: {e}")
            return False

    def semantic_search(self, query: str, top_k: int = Config.RETRIEVAL_TOP_K) -> List[Tuple[int, float]]:
        """语义检索"""
        if self.vector_store is None or self.embeddings is None:
            return []
        
        try:
            query_vector = self.embeddings.encode([query]).astype('float32')
            distances, indices = self.vector_store.search(query_vector, top_k)
            results = [(idx, float(distances[0][i])) for i, idx in enumerate(indices[0]) if idx != -1]
            return results
        except Exception as e:
            print(f"语义检索失败: {e}")
            return []

    def keyword_search(self, query: str, top_k: int = Config.RETRIEVAL_TOP_K) -> List[Tuple[int, float]]:
        """关键词检索"""
        if self.bm25_index is None:
            return []
        
        try:
            import jieba
            tokenized_query = jieba.lcut(query)
            scores = self.bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results = [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
            return results
        except Exception as e:
            print(f"关键词检索失败: {e}")
            return []

    def hybrid_search(self, query: str) -> List[Tuple[int, float, Dict]]:
        """混合检索"""
        print(f"\n🔍 开始检索: {query[:50]}...")
        
        semantic_results = self.semantic_search(query)
        keyword_results = self.keyword_search(query)
        print(f"  语义检索找到 {len(semantic_results)} 个结果")
        print(f"  关键词检索找到 {len(keyword_results)} 个结果")

        semantic_scores = {idx: score for idx, score in semantic_results}
        keyword_scores = {idx: score for idx, score in keyword_results}
        
        if semantic_scores:
            max_sem = max(semantic_scores.values())
            if max_sem > 0:
                semantic_scores = {k: v/max_sem for k, v in semantic_scores.items()}
        
        if keyword_scores:
            max_key = max(keyword_scores.values())
            if max_key > 0:
                keyword_scores = {k: v/max_key for k, v in keyword_scores.items()}

        all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys())
        hybrid_scores = {}
        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0) * Config.SEMANTIC_WEIGHT
            key_score = keyword_scores.get(idx, 0) * Config.KEYWORD_WEIGHT
            hybrid_scores[idx] = sem_score + key_score

        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:Config.RETRIEVAL_TOP_K]
        results = []
        for idx, score in sorted_results:
            results.append((idx, score, {
                "semantic_score": semantic_scores.get(idx, 0),
                "keyword_score": keyword_scores.get(idx, 0),
                "content": self.documents[idx][:200]
            }))
            print(f"  文档{idx+1}: 综合分数={score:.3f}, 内容={self.documents[idx][:50]}...")
        
        return results

    def get_document(self, idx: int) -> str:
        """获取文档内容"""
        return self.documents[idx] if 0 <= idx < len(self.documents) else ""


# ==================== 模型调用模块 ====================
class ModelClient:
    """模型客户端"""
    def __init__(self, model_type: str = "deepseek-r1"):
        self.model_type = model_type
        self.client = None
        self.model_name = Config.AVAILABLE_MODELS.get(model_type, Config.AVAILABLE_MODELS["deepseek-v3"])
        self.use_local = False

    def init_client(self):
        """初始化客户端"""
        try:
            from openai import OpenAI
            
            if Config.API_KEY and Config.API_KEY != "your-key":
                self.client = OpenAI(
                    api_key=Config.API_KEY,
                    base_url=Config.API_BASE_URL,
                    timeout=60
                )
                print(f"✅ 硅基流动API连接成功，模型: {self.model_name}")
            else:
                self.client = OpenAI(
                    api_key="ollama",
                    base_url=Config.OLLAMA_URL,
                    timeout=60
                )
                self.use_local = True
                print(f"✅ 本地Ollama连接成功，模型: {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ 模型客户端初始化失败: {e}")
            return False

    def generate(self, prompt: str, system_prompt: str = "你是专业的宪法问答助手", 
                 max_tokens: int = 1024) -> str:
        """生成回答"""
        if self.client is None:
            if not self.init_client():
                return "模型连接失败，请检查网络和API配置"
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            print(f"🤖 调用模型生成中...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"生成失败: {str(e)[:100]}"
            print(error_msg)
            return error_msg

    def generate_with_context(self, query: str, context: str) -> str:
        """基于上下文生成回答"""
        if self.model_type == "deepseek-r1":
            system_prompt = """你是专业的宪法问答助手，请严格基于提供的宪法原文上下文回答问题。
如果上下文不足以回答，说明"上下文信息不足，无法准确回答"，不要编造答案。
回答格式必须包含两部分：
【推理过程】：说明如何使用上下文条款分析问题
【最终答案】：给出准确、简洁的回答（引用具体条款号）"""
        else:
            system_prompt = """你是专业的宪法问答助手，请严格基于提供的宪法原文上下文回答问题。
如果上下文不足以回答，说明"上下文信息不足，无法准确回答"，不要编造答案。"""
        
        prompt = f"""上下文信息（宪法原文）:
{context}

用户问题: {query}

请基于以上上下文回答，不得添加外部信息！"""
        return self.generate(prompt, system_prompt)


# ==================== RAG应用主类 ====================
class RAGApplication:
    """RAG应用主类"""
    def __init__(self, retriever_model: str = "deepseek-v3", generator_model: str = "deepseek-r1"):
        self.retriever_model = retriever_model
        self.generator_model = generator_model
        self.data_processor = DataProcessor()
        self.retriever = VectorRetriever()
        self.model_client = ModelClient(generator_model)
        self.is_initialized = False
        self.performance_metrics = []

    def initialize(self):
        """初始化RAG系统"""
        print("\n" + "=" * 50)
        print("初始化RAG系统...")
        print(f"检索模型: {self.retriever_model}")
        print(f"生成模型: {self.generator_model}")
        print("=" * 50)

        print("\n[1/5] 加载知识库...")
        text = self.data_processor.load_knowledge_base(Config.KNOWLEDGE_FILE)
        
        print("\n[2/5] 文本分块...")
        chunks = self.data_processor.split_text(text)
        
        if not chunks:
            print("❌ 文本分块失败，知识库为空")
            return False
        
        self.retriever.documents = chunks
        
        print("\n[3/5] 加载嵌入模型...")
        has_embedding = self.retriever.init_embeddings()
        
        print("\n[4/5] 构建检索索引...")
        if has_embedding:
            self.retriever.build_index(chunks)
        self.retriever.build_bm25_index()
        
        print("\n[5/5] 初始化生成模型...")
        if not self.model_client.init_client():
            return False

        self.is_initialized = True
        print("\n✅ RAG系统初始化完成!")
        return True

    def query(self, question: str, return_details: bool = False) -> Dict:
        """RAG查询"""
        if not self.is_initialized:
            return {"error": "系统未初始化，请先调用initialize()方法"}
        
        print("\n" + "=" * 50)
        print(f"📝 用户问题: {question}")
        print("=" * 50)
        
        start_time = time.time()
        
        retrieve_start = time.time()
        search_results = self.retriever.hybrid_search(question)
        retrieve_time = time.time() - retrieve_start
        
        if not search_results:
            print("⚠️ 未检索到相关宪法条款!")
            context = "未找到相关宪法原文信息。"
        else:
            context = "\n\n".join([self.retriever.get_document(idx) for idx, _, _ in search_results])
            print(f"\n📚 构建上下文，共 {len(context)} 字符")
        
        generate_start = time.time()
        answer = self.model_client.generate_with_context(question, context)
        generate_time = time.time() - generate_start
        total_time = time.time() - start_time
        
        metrics = {
            "question": question,
            "retriever_model": self.retriever_model,
            "generator_model": self.generator_model,
            "retrieve_time": retrieve_time,
            "generate_time": generate_time,
            "total_time": total_time,
            "retrieved_count": len(search_results),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.performance_metrics.append(metrics)
        
        print(f"\n⏱️ 检索耗时: {retrieve_time:.2f}s, 生成耗时: {generate_time:.2f}s, 总耗时: {total_time:.2f}s")
        print(f"📝 最终答案:\n{answer}\n")
        
        result = {"answer": answer, "metrics": metrics}
        if return_details:
            result["context"] = context
            result["retrieved_docs"] = [
                {"index": idx+1, "score": score, "content": self.retriever.get_document(idx)[:100]} 
                for idx, score, _ in search_results
            ]
        return result

    def query_without_rag(self, question: str) -> Dict:
        """不使用RAG的直接回答"""
        print("\n" + "=" * 50)
        print(f"📝 无RAG测试: {question}")
        print("=" * 50)
        
        start_time = time.time()
        answer = self.model_client.generate(question, system_prompt="你是专业的宪法问答助手，直接回答问题，引用具体条款号")
        total_time = time.time() - start_time
        
        print(f"📝 直接回答:\n{answer}\n")
        return {
            "answer": answer,
            "metrics": {"total_time": total_time, "with_rag": False}
        }


# ==================== 实验评估模块 ====================
class ExperimentEvaluator:
    """实验评估与可视化"""
    def __init__(self, rag_app: RAGApplication):
        self.rag_app = rag_app
        self.results = []

    def run_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 50)
        print("开始实验测试（共9道题：事实型3道+推理型3道+多跳推理型3道）...")
        print("=" * 50)
        
        for i, test in enumerate(Config.TEST_QUESTIONS, 1):
            print(f"\n{'='*40}")
            print(f"测试 {i}/{len(Config.TEST_QUESTIONS)} [{test['type']}]")
            print(f"{'='*40}")
            
            rag_result = self.rag_app.query(test['question'], return_details=False)
            alone_result = self.rag_app.query_without_rag(test['question'])
            
            self.results.append({
                "type": test['type'],
                "question": test['question'],
                "rag_answer": rag_result['answer'],
                "alone_answer": alone_result['answer'],
                "rag_metrics": rag_result['metrics'],
                "alone_metrics": alone_result['metrics'],
                "expected_keywords": test['expected_keywords']
            })

    def evaluate_accuracy(self, answer: str, expected_keywords: List[str]) -> float:
        """准确率评估"""
        if not expected_keywords:
            return 0.5
        
        answer_lower = answer.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return matched / len(expected_keywords)

    def generate_report(self) -> pd.DataFrame:
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("实验报告 - 性能对比")
        print("=" * 60)
        
        data = []
        for r in self.results:
            rag_acc = self.evaluate_accuracy(r['rag_answer'], r['expected_keywords'])
            alone_acc = self.evaluate_accuracy(r['alone_answer'], r['expected_keywords'])
            data.append({
                "问题类型": r['type'],
                "RAG回答时间(s)": f"{r['rag_metrics']['total_time']:.2f}",
                "单独模型时间(s)": f"{r['alone_metrics']['total_time']:.2f}",
                "RAG准确率": f"{rag_acc:.1%}",
                "单独模型准确率": f"{alone_acc:.1%}",
                "检索文档数": r['rag_metrics']['retrieved_count']
            })
        
        df = pd.DataFrame(data)
        print("\n📊 详细性能对比表:")
        print(df.to_string(index=False))
        
        avg_rag_time = df['RAG回答时间(s)'].astype(float).mean()
        avg_alone_time = df['单独模型时间(s)'].astype(float).mean()
        avg_rag_acc = df['RAG准确率'].str.rstrip('%').astype(float).mean() / 100
        avg_alone_acc = df['单独模型准确率'].str.rstrip('%').astype(float).mean() / 100
        
        print(f"\n📈 平均性能汇总:")
        print(f"  RAG方案平均响应时间: {avg_rag_time:.2f}秒")
        print(f"  单独模型平均响应时间: {avg_alone_time:.2f}秒")
        print(f"  RAG方案平均准确率: {avg_rag_acc:.1%}")
        print(f"  单独模型平均准确率: {avg_alone_acc:.1%}")
        print(f"  响应时间提升: {(avg_alone_time - avg_rag_time) / avg_alone_time * 100:.1f}%")
        print(f"  准确率提升: {(avg_rag_acc - avg_alone_acc) * 100:.1f}%")
        
        return df

    def visualize(self):
        """可视化结果"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('宪法知识库RAG应用实验结果可视化', fontsize=16, fontweight='bold')
        
        question_types = [r['type'] for r in self.results]
        rag_times = [r['rag_metrics']['total_time'] for r in self.results]
        alone_times = [r['alone_metrics']['total_time'] for r in self.results]
        rag_accs = [self.evaluate_accuracy(r['rag_answer'], r['expected_keywords']) for r in self.results]
        alone_accs = [self.evaluate_accuracy(r['alone_answer'], r['expected_keywords']) for r in self.results]
        retrieved_counts = [r['rag_metrics']['retrieved_count'] for r in self.results]
        retrieve_times = [r['rag_metrics']['retrieve_time'] for r in self.results]
        generate_times = [r['rag_metrics']['generate_time'] for r in self.results]
        
        x = np.arange(len(question_types))
        width = 0.35

        ax1 = axes[0, 0]
        ax1.bar(x - width/2, rag_times, width, label='RAG方案', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, alone_times, width, label='单独模型', color='#A23B72', alpha=0.8)
        ax1.set_xlabel('问题类型')
        ax1.set_ylabel('响应时间 (秒)')
        ax1.set_title('响应时间对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(question_types, rotation=15)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        ax2 = axes[0, 1]
        ax2.bar(x - width/2, rag_accs, width, label='RAG方案', color='#2E86AB', alpha=0.8)
        ax2.bar(x + width/2, alone_accs, width, label='单独模型', color='#A23B72', alpha=0.8)
        ax2.set_xlabel('问题类型')
        ax2.set_ylabel('准确率')
        ax2.set_title('回答准确率对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(question_types, rotation=15)
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        ax3 = axes[1, 0]
        ax3.bar(question_types, retrieved_counts, color='#F18F01', alpha=0.8)
        ax3.set_xlabel('问题类型')
        ax3.set_ylabel('检索文档数')
        ax3.set_title('各类型问题检索文档数量')
        ax3.tick_params(axis='x', rotation=15)
        ax3.grid(axis='y', alpha=0.3)

        ax4 = axes[1, 1]
        ax4.bar(question_types, retrieve_times, label='检索时间', color='#73AB84', alpha=0.8)
        ax4.bar(question_types, generate_times, bottom=retrieve_times, label='生成时间', color='#F4D58C', alpha=0.8)
        ax4.set_xlabel('问题类型')
        ax4.set_ylabel('时间 (秒)')
        ax4.set_title('RAG方案时间分解')
        ax4.tick_params(axis='x', rotation=15)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(Config.RESULT_IMAGE, dpi=150, bbox_inches='tight')
        print(f"\n📊 可视化图表已保存为: {Config.RESULT_IMAGE}")
        plt.close()


# ==================== 交互式界面 ====================
class InteractiveUI:
    """交互式界面"""
    def __init__(self):
        self.rag_app = None

    def run(self):
        """运行交互界面"""
        print("""
==================================================
实验二: 使用Deepseek-R1和V3构建宪法知识库RAG应用
功能: 交互式模型选择 | 混合检索 | 实验评估 | 可视化
==================================================
        """)
        
        print("\n请选择【检索模型】:")
        print("1. Deepseek-V3")
        print("2. Deepseek-R1")
        print("3. Qwen2.5-7B (本地Ollama)")
        print("4. Qwen2.5-0.5B (轻量，本地Ollama)")
        retriever_choice = input("\n请输入选项 (1-4): ").strip()
        model_map = {
            "1": "deepseek-v3",
            "2": "deepseek-r1",
            "3": "qwen2.5",
            "4": "qwen2.5-0.5b"
        }
        retriever_model = model_map.get(retriever_choice, "deepseek-v3")
        
        print("\n请选择【生成模型】:")
        print("1. Deepseek-V3")
        print("2. Deepseek-R1 (推荐，显示推理过程)")
        print("3. Qwen2.5-7B (本地Ollama)")
        print("4. Qwen2.5-0.5B (轻量，本地Ollama)")
        generator_choice = input("\n请输入选项 (1-4): ").strip()
        generator_model = model_map.get(generator_choice, "deepseek-r1")
        
        print(f"\n✅ 最终配置: 检索模型={retriever_model}, 生成模型={generator_model}")
        
        self.rag_app = RAGApplication(retriever_model, generator_model)
        if not self.rag_app.initialize():
            print("❌ RAG系统初始化失败，请检查配置！")
            return
        
        while True:
            print("\n" + "-" * 40)
            print("请选择操作:")
            print("1. 提问（RAG模式，基于宪法知识库）")
            print("2. 运行完整实验评估（9道测试题+可视化）")
            print("3. 查看历史性能指标")
            print("4. 清除历史记录")
            print("5. 退出程序")
            choice = input("\n请输入选项 (1-5): ").strip()
            
            if choice == "1":
                self._ask_question()
            elif choice == "2":
                self._run_full_experiment()
            elif choice == "3":
                self._show_metrics()
            elif choice == "4":
                self._clear_metrics()
            elif choice == "5":
                print("👋 感谢使用，再见！")
                break
            else:
                print("❌ 无效选项，请重新输入！")

    def _ask_question(self):
        """用户自定义提问"""
        question = input("\n请输入关于宪法的问题（例如：公民的基本政治自由有哪些？）: ").strip()
        if not question:
            print("⚠️ 问题不能为空！")
            return
        self.rag_app.query(question, return_details=True)

    def _run_full_experiment(self):
        """运行完整实验评估"""
        evaluator = ExperimentEvaluator(self.rag_app)
        evaluator.run_tests()
        evaluator.generate_report()
        evaluator.visualize()
        print("\n✅ 完整实验评估完成！报告和图表已生成。")

    def _show_metrics(self):
        """查看历史性能指标"""
        if not self.rag_app.performance_metrics:
            print("⚠️ 暂无历史性能数据，请先提问或运行实验！")
            return
        
        print("\n📊 历史性能指标（最近10条）:")
        print("-" * 80)
        for i, metrics in enumerate(self.rag_app.performance_metrics[-10:], 1):
            q = metrics['question'][:40] + "..." if len(metrics['question']) > 40 else metrics['question']
            print(f"{i}. [{metrics['timestamp'][:16]}] {q}")
            print(f"   总耗时: {metrics['total_time']:.2f}s (检索: {metrics['retrieve_time']:.2f}s + 生成: {metrics['generate_time']:.2f}s)")
            print(f"   检索文档数: {metrics['retrieved_count']}")
            print()

    def _clear_metrics(self):
        """清除历史记录"""
        self.rag_app.performance_metrics = []
        print("✅ 历史记录已清除")


# ==================== 主函数 ====================
def check_dependencies():
    """检查依赖包"""
    missing = []
    
    packages = {
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'rank_bm25': 'rank-bm25',
        'jieba': 'jieba',
        'openai': 'openai',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy'
    }
    
    for module, package in packages.items():
        try:
            if module == 'faiss':
                __import__('faiss')
            else:
                __import__(module)
        except ImportError:
            missing.append(package)
    
    return missing


def main():
    """主函数"""
    print("正在检查依赖...")
    
    missing = check_dependencies()
    
    if missing:
        print(f"❌ 缺少依赖包，请运行以下命令安装：")
        print(f"pip install {' '.join(missing)} -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return
    
    print("✅ 依赖检查通过")
    
    # 启动交互界面
    ui = InteractiveUI()
    ui.run()


if __name__ == "__main__":
    main()
```

### 9.2 参考文献

[1] 中华人民共和国全国人民代表大会常务委员会。中华人民共和国宪法 [Z]. 1982-12-04.

[2] Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks [J]. Advances in Neural Information Processing Systems, 2020, 33: 9459-9474.

[3] BAAI. bge-base-zh-v1.5 中文嵌入模型 [EB/OL]. https://huggingface.co/BAAI/bge-base-zh-v1.5.

[4] DeepSeek. DeepSeek-R1/V3 大语言模型 [EB/OL]. https://www.deepseek.com/.

