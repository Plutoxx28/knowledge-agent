
## **一、检索增强生成（RAG）概览**
![[how-rag-works.png]]
### **A. RAG的定义与核心价值**

RAG是一种通过外部数据增强LLM能力的技术，它使模型能够接触到其训练数据之外的、通常是特定领域或实时更新的信息 2。其核心价值在于，RAG能够使LLM在生成答案时参考权威知识库，从而提供更准确、更可靠且与特定上下文相关的输出 7。这种方法不仅降低了模型产生幻觉的风险，还使得LLM能够应用于需要高度事实准确性和最新信息的场景，如企业内部知识问答、客户支持和实时信息查询 8。与需要大量计算资源和时间进行模型微调或重新训练相比，RAG提供了一种更经济高效地将新知识整合到LLM中的途径 2。

### **B. RAG系统的核心组件与通用工作流程**

1、一个典型的RAG系统无论具体实现如何，通常都包含以下核心组件和能力

- 数据连接器（Connectors）：负责连接各种企业数据源（如数据库、对象存储、代码库、SaaS平台）与向量数据库。
- 数据处理（Data Processing）：对来自不同来源（如PDF、Word文档、网页）的原始数据进行提取、清洗、格式化和分块（Chunking），为索引做好准备。
- 嵌入模型（Embeddings）：将处理后的文本数据（文档块和用户查询）转换为数值向量表示，即嵌入向量，以捕捉其语义含义。
- 向量数据库（Vector Database）：存储文本块的嵌入向量及其元数据，并优化用于高效的相似性搜索和检索。
- 检索器（Retriever）：根据用户查询的嵌入向量，在向量数据库中搜索并取回最相关的文本块作为上下文。
- 大型语言模型（Foundation Model / LLM）：接收用户查询和检索到的上下文，并基于这些信息生成最终的、更准确的回应。
- 编排器（Orchestrator）：负责调度和管理整个端到端的工作流程，从接收查询到生成回应。
- 用户体验（User Experience）：通常是一个对话式聊天界面，提供聊天历史记录、用户反馈等功能。
- 安全护栏（Guardrails）：确保查询、提示、检索到的上下文以及LLM的回应是准确、负责任、合乎道德且无偏见的。
- 身份与用户管理（Identity and User Management）：控制用户对应用程序的访问权限。

2、RAG的工作流程大致可以分为两个阶段：数据准备（一次性）和查询处理（按需多次）。

数据准备阶段（离线）**：

- 数据加载与处理：通过数据连接器从各种来源加载数据，并进行清洗、格式化等预处理。
    
- 文本分块：将长文档分割成较小的、语义相关的文本块。
    
- 嵌入生成与存储：使用嵌入模型为每个文本块生成嵌入向量，并将这些向量连同原始文本和元数据一起存储到向量数据库中，创建索引。
    

查询处理阶段（在线）：

- 用户查询：用户以自然语言提交查询。
    
- 查询嵌入：使用相同的嵌入模型将用户查询转换为嵌入向量。
    
- 相似性搜索与上下文检索：编排器在向量数据库中执行相似性搜索，找出与查询向量最相关的文本块。
    
- 提示增强：编排器将检索到的文本块（上下文）与原始用户查询整合成一个新的提示（Augmented Prompt）。
    
- LLM生成回应：编排器将增强后的提示发送给LLM，LLM利用提供的上下文生成回应。
    
- 后处理与呈现：生成的回应可能经过安全护栏等模块处理后，最终呈现给用户。
    

这个流程使得LLM能够利用外部知识库的实时信息，生成更准确、更可靠的答案 7。

## **二、构建RAG系统的关键考量因素**
![[encoder-decode.png]]
成功构建一个高效的RAG系统需要对多个方面进行细致的规划和优化。

### **A. 数据准备**

数据准备是RAG系统的基石，其质量直接影响后续检索和生成的效果。

1、文档加载与解析 RAG系统的第一步是从各种数据源加载文档。这些数据源可能包括PDF、Word文档、网页、数据库记录等 5。加载后，需要对文档进行解析，提取纯文本内容，并处理可能存在的格式问题，如特殊字符、不规则布局等 12。例如，RAGFlow支持多种文件格式，包括DOCX、XLSX、PDF、TXT、图片（JPEG、JPG、PNG、TIF、GIF）、CSV、JSON、EML和HTML等 11。
    
2、分块策略（Chunking Strategies） 由于LLM的上下文窗口限制以及为了提高检索效率和准确性，长文档需要被分割成更小的、语义完整的单元，即“块”（Chunks）7。选择合适的分块策略至关重要。

- 固定大小分块（Fixed-size Chunking）：这是最简单的方法，将文本按固定字符数或Token数分割，通常会设置块之间的重叠（Overlap）以保持上下文连续性 16。虽然简单，但可能切断语义单元 13。

- 递归分块（Recursive Chunking）：使用一系列分隔符（如段落、句子、单词）按层级递归地分割文本，直到块大小符合要求。这种方法试图保持语义单元的完整性 16。LangChain的RecursiveCharacterTextSplitter是一个常用工具 13。

- 语义分块（Semantic Chunking）：基于文本的语义相似性进行分割，将语义相近的句子组合在一起。这通常需要先对句子进行嵌入，然后根据嵌入向量的相似度来确定分割点 13。LangChain的SemanticChunker利用嵌入模型实现此功能 13。

```
### 原理

传统分块方法按固定字符数切分文档，可能割裂语义单元。语义分块根据内容的语义边界(如段落、章节、主题)进行分割，保持内容的语义完整性。

### 实现方法

python

```python
def semantic_chunking(self, text: str) -> List[str]:
    """基于语义边界进行文档分块"""
    # 1. 首先按明显的结构边界分割
    sections = []
    
    # 按标题模式分割
    heading_pattern = r'(?m)^(#{1,6}|\d+(\.\d+)*\s+)[A-Za-z0-9]'
    section_texts = re.split(heading_pattern, text)
    
    # 处理每个初步分割的部分
    for section_text in section_texts:
        if not section_text.strip():
            continue
            
        # 2. 检查长度，如果过长则进一步分割
        if len(section_text) > self.max_chunk_size:
            # 按段落分割
            paragraphs = re.split(r'\n\s*\n', section_text)
            
            current_chunk = ""
            for paragraph in paragraphs:
                # 如果添加这个段落会超过大小限制，先保存当前块
                if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                    sections.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                        
            # 添加最后一个块
            if current_chunk:
                sections.append(current_chunk.strip())
        else:
            sections.append(section_text.strip())
    
    # 3. 确保每个块都有足够的上下文
    enhanced_sections = []
    for i, section in enumerate(sections):
        # 添加块标识和可能的上下文信息
        metadata = self._extract_context_info(section, i, sections)
        enhanced_section = f"{metadata}\n\n{section}"
        enhanced_sections.append(enhanced_section)
    
    return enhanced_sections
```

```
高级实现：使用LLM辅助分块

python

```python
def llm_assisted_chunking(self, text: str) -> List[str]:
    """使用LLM识别语义边界进行分块"""
    # 先进行初步分割，以减少LLM处理量
    rough_chunks = self._split_by_length(text, 2000)
    
    final_chunks = []
    for rough_chunk in rough_chunks:
        # 让LLM识别语义单元并标记分割点
        prompt = f"""
        分析以下文本，识别其中的自然语义边界，如主题转换、章节结束等。
        在每个应该分割的点插入[SPLIT]标记。保持每个分割后的部分在300-1000字之间。
        保留原文的所有内容，只添加[SPLIT]标记。
        
        文本: "{rough_chunk}"
        """
        
        marked_text = self.llm.invoke(prompt)
        
        # 根据[SPLIT]标记分割
        semantic_chunks = marked_text.split("[SPLIT]")
        final_chunks.extend([chunk.strip() for chunk in semantic_chunks if chunk.strip()])
    
    return final_chunks
```
```
应用效果

- **提高检索质量**：保持语义完整性，减少上下文断裂
- **减少重复内容**：自然边界分割减少了内容重叠
- **改善回答准确性**：LLM获得完整的语义单元，生成更连贯的回答
```

- 内容感知分块（Content-aware/Document-based Chunking）：利用文档的固有结构（如Markdown的标题、HTML标签、代码中的函数或类）进行分块 16。例如，RAGFlow提供了基于文档布局和格式的多种“模板化分块”方法，如针对论文、书籍、法律文件、演示文稿等的特定模板 14。这种方法能更好地保留文档的逻辑结构。例如，MarkdownHeaderTextSplitter可以根据Markdown标题进行分割 13。

- 代理分块（Agentic Chunking）：这是一种更高级的实验性方法，利用LLM本身来决定如何根据语义和内容结构（如段落类型、章节标题、步骤说明）进行分块，模拟人类处理长文档时的推理过程 16。这旨在创建更具上下文感知能力的块。有研究提出基于Phi-2小型语言模型（SLM）的RAG系统，利用前瞻性语义分块（forward-looking semantic chunking）根据嵌入相似性自适应确定解析断点，以有效处理不同格式的文档 22。

选择分块策略时，需要考虑块大小（Chunk Size）和块重叠（Chunk Overlap）。块大小应平衡上下文的完整性和检索的精确性，通常需要实验确定 15。块重叠有助于在块的边界处保留上下文信息 16。RAGFlow允许用户查看分块结果并进行手动干预，如添加关键词以提高特定块的检索排名 14。


3、元数据提取与利用 提取并利用文档元数据（如标题、作者、创建日期、类别、关键词、章节信息等）对于提升RAG系统的检索性能至关重要 13。元数据可以：
    
增强检索精度：允许在检索时基于元数据进行过滤和排序，从而缩小搜索范围，提高返回结果的相关性 25。
        
改善上下文相关性：为LLM提供关于文档来源、用途或主题领域的额外上下文，帮助生成更准确、连贯的回应 26。
        
实现个性化：通过标记用户过去的交互和偏好，系统可以定制回应 26。
        
提高可解释性：提供关于文档的上下文信息，有助于解释AI的输出 23。 有效的元数据模式设计应与应用领域的特定需求对齐 26。在向量数据库中将元数据与块一起存储，可以在检索时实现基于元数据的过滤 13。
        
4、嵌入生成（Embedding Generation） 嵌入是将文本块转换为捕捉其语义含义的数值向量的过程。选择合适的嵌入模型对RAG系统的性能有显著影响 27。
    
    - **模型选择标准**：
        
        - **性能（MTEB得分）**：Massive Text Embedding Benchmark (MTEB) 是一个广泛认可的基准，用于评估嵌入模型在各种任务（包括检索）上的表现 27。检索平均分（Retrieval Average）是RAG场景下的一个重要参考指标 30。
            
        - **上下文窗口**：模型能处理的最大Token数。较长的上下文窗口有助于处理更长的文本片段 27。例如，OpenAI的text-embedding-ada-002支持8192个Token 27。
            
        - **维度（Dimensionality）**：嵌入向量的维度。更高维度可能捕捉更细致的语义，但也需要更多计算资源和存储空间 27。较低维度则更快更高效，但可能损失部分语义丰富性。一些模型如OpenAI的text-embedding-3-small和Nomic的Embed v1.5采用Matryoshka Representation Learning (MRL) 技术，即使在较低维度（如256）也能保持良好性能 29。
            
        - **训练数据与领域特异性**：通用模型（如OpenAI系列、BGE系列）通常在大量通用文本上训练。对于特定领域（如法律、医疗、金融），使用在该领域数据上训练或微调的嵌入模型（如LegalBERT, BioBERT, 或针对金融报告微调的模型）通常能获得更好的性能 27。
            
        - **成本**：API模型（如OpenAI, Cohere）按使用量收费，而开源模型（如BGE, GTE, Jina）可本地部署，成本主要为计算资源 27。OpenAI的text-embedding-3-large每百万Token约0.13美元，text-embedding-3-small约0.02美元 27。
            
        - **计算资源需求**：模型大小（参数量）和维度影响本地部署所需的内存和计算能力 30。
            
        - **兼容性与集成**：确保所选模型与RAG框架和向量数据库兼容。
            
    - **流行的嵌入模型**：
        
        - **OpenAI系列**：text-embedding-3-large 和 text-embedding-3-small 是目前性能较好的商业模型 27。text-embedding-ada-002 曾是广泛使用的型号 27。
            
        - **开源模型**：BGE (BAAI General Embedding) 系列（如 bge-large-en-v1.5） 33、Nomic Embed 33、Jina Embeddings 27、E5系列 34、SFR-Embedding 34、GTE系列 34 都是MTEB排行榜上表现优异的开源选项。NVIDIA的NV-Embed-v2具有32768 Token的长上下文窗口，适合长文档 27。
            
        - **特定数据类型模型**：如jina-embeddings-v2-base-code用于代码嵌入 29。SPLADE等稀疏向量模型在处理特定术语（如医学领域）时有优势 29。
            
    - **嵌入生成与管理策略**：
        
        - **批量处理（Batching）**：在生成嵌入时，对文本块进行批量处理可以提高效率，尤其是在使用GPU加速时 37。
            
        - **硬件加速**：利用GPU甚至专门的AI加速硬件（如FPGA）可以显著加快嵌入生成和检索速度 37。
            
        - **异步更新与增量嵌入**：对于动态变化的数据源，需要定期更新嵌入。可以采用完全重新嵌入、增量嵌入（仅更新变化部分）或混合方法 2。异步更新文档和嵌入表示，可以通过自动化实时流程或定期批处理完成 2。
            
        - **缓存**：缓存嵌入结果可以避免重复计算，节省成本和时间 41。
            
        - **版本控制**：对嵌入模型和生成的嵌入进行版本控制，有助于跟踪性能变化和在必要时回滚 39。
            
5、向量数据库（Vector Databases） 向量数据库专门用于存储和高效查询高维嵌入向量 5。选择向量数据库时需考虑以下因素：
    
    - **可扩展性（Scalability）**：处理大规模向量数据集（从数百万到数十亿）的能力 42。
        
    - **查询性能（Query Performance）**：低延迟（如亚100毫秒）和高吞吐量的查询能力 42。
        
    - **易用性与维护**：部署、管理和集成的复杂性。托管服务（如Pinecone, Weaviate Cloud, Zilliz Cloud）通常运营复杂度较低 42。
        
    - **成本**：开源自托管（如Milvus, Weaviate, Qdrant, Chroma）与商业托管服务的成本模型不同，前者主要是基础设施成本，后者通常基于存储、查询量或计算资源付费 42。
        
    - **特性支持**：
        
        - **元数据过滤**：在向量搜索的同时根据元数据属性进行过滤，对RAG至关重要 42。
            
        - **混合搜索（Hybrid Search）**：结合向量相似性搜索和传统关键词搜索（如BM25）的能力，可以提高检索的鲁棒性 42。
            
        - **索引类型**：支持多种索引算法（如HNSW, IVF），以适应不同性能需求 42。
            
        - **数据一致性**：例如，Vertex AI Vector Search是最终一致性，而RagManagedDb提供强一致性 44。
            
        - **多租户**：支持在单个数据库实例中隔离不同用户或应用的数据 46。
            
    - **开源与商业选项**：
        
        - **开源/可自托管**：Milvus 42, Weaviate 42, Qdrant 42, Chroma 42, FAISS (库) 46, PGVector (Postgres扩展) 46。
            
        - **商业/托管服务**：Pinecone 42, Zilliz Cloud (Milvus托管) 43, Vertex AI Vector Search 44, Elasticsearch 46, OpenSearch 46, Astra DB 48。
            
        - RAGFlow默认使用Elasticsearch，也支持切换到Infinity作为文档引擎 20。
            

下表总结了一些流行向量数据库的特性对比，信息主要来源于 42 和 44：

|向量数据库|类型|典型查询时间 (1M向量)|最大规模|扩展模型|运营复杂度|混合搜索|元数据过滤|
|---|---|---|---|---|---|---|---|
|Pinecone|托管服务|20-80ms|数十亿|自动|低|支持|支持|
|Weaviate|开源/托管|30-150ms|数亿|手动/Kubernetes|中|内置BM25|支持|
|Chroma|开源|50-200ms|数百万|手动|低|有限|支持|
|Milvus|开源/托管|10-100ms|数十亿+|手动/Kubernetes|高|支持|支持|
|Vertex AI Vector Search|托管服务 (Google)|N/A|可扩展|Google Cloud|中|支持|支持|
|Elasticsearch|开源/托管|N/A|可扩展|手动/托管|中-高|支持|支持|
|PGVector|Postgres扩展|N/A|取决于Postgres规模|Postgres|中|有限|支持|
|RAGFlow ManagedDB|托管服务 (RAGFlow)|N/A|有限 (不推荐大规模)|RAGFlow|低|KNN|N/A|

选择向量数据库时，应基于预期的规模、性能需求、运营能力和预算进行综合评估 42, 43。

### **B. 检索机制**

检索机制负责根据用户查询从向量数据库中高效地提取最相关的上下文信息。

1、**稀疏、稠密与混合检索**

**稀疏检索（Sparse Retrieval）**：主要基于关键词匹配，如TF-IDF或BM25算法。它们计算查询和文档之间共享词汇的加权重复度。这类方法简单、高效，无需训练，不考虑词序和句法，但无法处理词汇表外（OOV）的词或理解同义词 51。Haystack提供了ElasticsearchBM25Retriever等实现 51。

```
### 基本原理

稠密检索使用神经网络模型(如BGE、E5等)将文本转换为**固定维度的稠密向量**，其中每个维度都有一个连续的值。这些向量通常是低维的(如768或1024维)，但信息"稠密"地分布在每个维度上。

### 工作过程

1. **向量生成**：
    
    ```
    文档/查询 → Embedding模型 → 稠密向量(例如[0.2, -0.5, 0.7, ...])
    ```
    
2. **相似度计算**：
    - 通常使用余弦相似度或点积计算两个向量的相似程度
    - 查询向量与所有文档向量进行比较，返回最相似的结果

### 特点

- **捕捉语义相似性**：能够理解同义词、上下文和概念关系
- **维度固定**：无论文本长度如何，生成的向量维度相同
- **处理新词能力**：可以处理训练数据中未见过的词语
- **需要神经网络**：计算密集型，通常需要GPU加速

### 示例

当用户搜索"如何治疗感冒"，稠密检索也可以找到谈论"感冒药物使用方法"的文档，即使它们没有共同的关键词。
```

 **稠密检索（Dense Retrieval）**：利用嵌入向量捕捉文本的语义含义。查询和文档都被转换为向量，通过计算向量间的相似度（如余弦相似度、点积）来找到相关文档。这类方法功能强大，能理解语义相似性，但计算成本通常高于稀疏检索，且需要训练好的嵌入模型 29。Haystack支持多种基于嵌入的检索器，如AstraEmbeddingRetriever、ChromaEmbeddingRetriever等 51。

```
### 基本原理

稀疏检索基于传统的词频统计方法，将文本表示为**高维稀疏向量**，其中大多数维度的值为0。每个维度通常对应词汇表中的一个词，值代表该词的重要性。

### 工作过程

1. **向量生成**：
    
    ```
    文档/查询 → 词频统计(如BM25) → 稀疏向量([词1:权重, 词2:权重, ...])
    ```
    
2. **相似度计算**：
    - 通常基于关键词匹配和统计权重(如TF-IDF或BM25算法)
    - 只比较两个文本中共同出现的词汇

### 特点

- **精确关键词匹配**：在精确匹配关键词上表现出色
- **维度可变**：维度等于词汇表大小，通常很高(如数万维)
- **计算高效**：大多数值为0，计算稀疏矩阵高效
- **易于解释**：可以清楚看到哪些关键词导致了匹配

### 示例

当用户搜索"RAG系统优化"，稀疏检索能精确找到包含这些确切关键词的文档。
```

**稀疏、稠密case示例**

```
## 两种方法的直观比较

让我们通过一个简化的例子来理解两种检索方法的区别：

假设我们有一个简单的文档集合：

1. "如何治疗感冒和发烧"
2. "常见感冒药物的使用方法"
3. "感冒与流感的区别"

当用户查询"感冒药推荐"时：

### 稀疏检索(BM25)如何工作

1. 分析查询中的关键词："感冒"和"药"
2. 为每个文档计算BM25分数：
    - 文档1包含"感冒"，但没有"药"
    - 文档2包含"感冒"和"药物"("药物"与"药"相关)
    - 文档3只包含"感冒"
3. 因为文档2匹配了最多的关键词，所以它的排名最高

### 稠密检索(如BGE)如何工作

1. 将查询"感冒药推荐"转换为稠密向量
2. 计算查询向量与每个文档向量的相似度：
    - 文档1讨论"治疗感冒"，语义上与药物推荐相关
    - 文档2直接谈论"感冒药物"，语义匹配度很高
    - 文档3讨论"感冒"但不涉及药物
3. 文档2和文档1可能都会有较高的相似度得分，但文档2可能排名更高
```

 **混合检索（Hybrid Search）**：结合稀疏检索和稠密检索的优势，以提高检索的鲁棒性和准确性。例如，可以同时使用BM25和嵌入向量进行检索，然后融合两者的结果。这种方法对于既包含特定关键词又涉及复杂语义的查询特别有效 42。Haystack的DocumentJoiner组件和QdrantHybridRetriever支持混合检索 51。RAGFlow也采用加权关键词相似度和加权向量余弦相似度的组合进行检索 53。

 **不同模型和策略的优缺点对比：**

| 模型/策略             | 优点           | 缺点          | 适用场景       |
| ----------------- | ------------ | ----------- | ---------- |
| OpenAI Embeddings | 集成简单，性能稳定    | 需要API付费，有延迟 | 快速开发，追求稳定性 |
| E5-large-v2       | 多语言支持优秀，开源免费 | 计算资源要求高     | 多语言文档，英文为主 |
| BGE-large-zh      | 中文效果优异，开源免费  | 英文可能不如E5    | 中文文档为主的系统  |
| 稠密检索              | 理解语义相似性      | 可能漏掉关键词精确匹配 | 需要语义理解的查询  |
| 稀疏检索(BM25)        | 精确捕获关键词      | 无法理解语义相关性   | 精确关键词查询    |
| Hybrid混合检索        | 兼顾语义和关键词     | 实现复杂，需要调参   | 追求最高检索质量   |

2、**高级检索技术**
    
**重新排序（Re-ranking）**：在初步检索获得候选文档后，使用更复杂的模型（如交叉编码器 Cross-encoders）对这些文档进行重新排序，以提高最终结果的相关性。交叉编码器能够更细致地比较查询和每个候选文档的语义相关性，但计算成本较高，因此通常只用于对少量候选文档进行重排。RAGFlow支持可选的重排模型，但提示使用会显著增加响应时间。
        
**多查询检索（Multi-query Retrieval）**：通过LLM从原始用户查询生成多个变体查询（如同义词替换、不同措辞），然后对每个变体查询执行检索，并将结果合并。这有助于召回更广泛的相关文档，特别是当原始查询表述不佳或存在歧义时 41。

```
### 原理

多查询扩展通过LLM生成原始查询的多个变体，从不同角度探索问题，扩大检索范围，提高召回率。

### 实现方法

python

```python
def generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
    """使用LLM生成原始查询的变体"""
    prompt = f"""
    我需要从不同角度理解用户的问题。
    原始问题: "{query}"
    
    请生成{num_variations}个不同表达但语义相近的问题变体，它们应该:
    1. 使用不同的词汇和表达方式
    2. 考虑不同的专业术语和同义词
    3. 改变问题结构但保持原意
    
    直接返回这些变体，每行一个，不要有编号或额外说明。
    """
    
    response = self.llm.invoke(prompt)
    variations = [line.strip() for line in response.strip().split('\n') if line.strip()]
    
    # 确保返回正确数量的变体
    variations = variations[:num_variations]
    
    # 合并原始查询和变体
    all_queries = [query] + variations
    
    return all_queries

def multi_query_search(self, query: str, k: int = 4):
    """执行多查询扩展检索"""
    # 生成查询变体
    query_variations = self.generate_query_variations(query)
    
    # 为每个查询变体检索文档
    all_docs = []
    seen_doc_ids = set()
    
    for query_variant in query_variations:
        # 为每个变体执行检索
        docs = self.vector_store.similarity_search(query_variant, k=k//2)
        
        # 添加未重复的文档
        for doc in docs:
            doc_id = self._get_doc_id(doc)
            if doc_id not in seen_doc_ids:
                all_docs.append(doc)
                seen_doc_ids.add(doc_id)
    
    # 对合并后的结果重新排序（可选）
    if hasattr(self, 'rerank_documents'):
        all_docs = self.rerank_documents(query, all_docs)
    
    return all_docs[:k]
```
```
### 应用效果

- **改进召回率**：可提高20-30%的相关文档召回
- **处理同义表达**：捕获用户可能使用的不同表达方式
- **实现成本低**：仅需1-2个额外LLM调用

例如，原始查询"RAG系统优化方法"可能生成变体：

- "如何提高检索增强生成系统的性能"
- "RAG架构的效率提升技术"
- "增强检索增强生成模型效果的策略"
```

**假设性文档嵌入（Hypothetical Document Embeddings, HyDE）**：首先让LLM根据用户查询生成一个“假设性”的、回答该查询的文档，然后将这个假设性文档嵌入，并用其嵌入向量去检索真实的相似文档。这种方法旨在通过生成与答案更相似的查询嵌入来提高语义匹配度 52。
        
**句子窗口检索（Sentence Window Retrieval）**：索引文档时将其分解为句子或小块。检索时，首先找到与查询最相关的句子，然后检索这些句子周围的上下文（如前后几个句子或包含这些句子的更大文本块），以提供更完整的语境 52。
        
**自动合并检索/父文档检索（Auto-merging Retrieval / Parent Document Retriever）**：基于文档的层级结构。先检索最相关的叶节点（小块），然后根据一定条件（如多个叶节点同属一个父节点）决定是返回这些叶节点还是它们的父节点（更大的块或整个文档）。这有助于在提供详细信息和保持宏观上下文之间取得平衡 41。
        
**知识图谱检索（Knowledge Graph Retrieval）**：如果数据中存在结构化的实体和关系信息，可以构建知识图谱。在检索时，除了向量搜索外，还可以利用图谱中的连接关系进行多跳问答，探索实体间的关联 54。RAGFlow支持在检索中利用知识图谱进行多跳问答，但这会增加检索时间 54。
        
**融合检索（Fusion Retrieval）**：例如RRF (Reciprocal Rank Fusion) 算法，可以合并来自不同检索方法（如BM25和向量检索）的排序列表，对在多个列表中都排名靠前的文档给予更高分数 56。RAGFlow的“多路召回与融合重排”也体现了这一思想 59。

**自动上下文精简**：检索到的文档常包含冗余或不相关内容，自动精简可去除冗余，突出核心信息，优化token使用。选择和组合这些检索策略需要根据具体应用场景、数据特性和性能要求进行权衡。

### **C. 生成与上下文增强**

在检索到相关信息后，下一步是利用这些信息来增强LLM的生成能力，使其产生更准确、更有依据的回应。

1. 提示工程（Prompt Engineering） 提示工程是指导LLM如何利用检索到的上下文来回答问题的关键。一个精心设计的提示模板可以将用户查询和检索到的文档块有效地整合起来，引导LLM生成期望的输出 2。
    
    - **结构化提示**：提示通常包含指令、用户问题以及检索到的上下文。例如，Haystack的PromptBuilder组件使用Jinja2模板语言，允许将检索到的文档（documents）和用户查询（query）作为变量填入模板中 61。RAGFlow的Generate组件也使用系统提示和键（变量）来指定LLM的数据输入 64。
        
    - **明确指示**：清晰地指示LLM基于提供的上下文回答问题，并可以要求其进行引用、总结或避免编造信息 65。例如，可以指示模型“仅使用提供的上下文回答问题，如果上下文中没有答案，请说明不知道”。
        
    - **上下文格式化**：检索到的多个文档块需要以LLM易于理解的方式组织在提示中，例如，通过换行符或特定标记分隔每个文档块 10。
        
    - **迭代优化**：提示设计往往是一个迭代过程，需要根据生成结果不断调整和优化提示的措辞和结构。
        
2. 利用检索到的上下文 LLM利用增强后的提示中的上下文信息来生成回应。上下文的质量和相关性直接影响生成结果的质量。
    
    - **上下文长度管理**：LLM有上下文窗口限制。如果检索到的上下文过长，可能需要进行截断、总结或选择最重要的部分 3。一些研究表明，将最重要的信息放在上下文的开头或结尾可能有助于LLM更好地利用它们（“Lost in the Middle”现象）68。
        
    - **引用来源（Citations）**：为了提高透明度和可信度，RAG系统可以设计为在生成的回应中包含对所用信息来源的引用 71。RAGFlow强调其“有根据的引用”功能，允许快速查看关键参考文献和可追溯的引文，以支持有根据的答案 20。RAGFlow的Generate组件有一个“Cite”开关，用于设置是否引用原始文本作为参考 64。
        
    - **处理不相关或噪声上下文**：检索到的上下文可能包含不相关甚至错误的信息。LLM需要具备一定的鲁棒性来处理这些噪声。一些高级提示技术，如思维链（Chain-of-Thought, CoT）的变种，如思维线索（Thread of Thought, ThoT）或笔记链（Chain-of-Note, CoN），旨在帮助LLM更好地分析和筛选上下文信息 65。CoN通过为检索到的文档生成阅读笔记来评估其相关性，并整合信息形成最终答案，从而有效滤除不相关内容 65。
        
3. 减少幻觉（Reducing Hallucinations） 幻觉是指LLM生成看似合理但实际上不正确或无事实依据的信息。RAG本身通过提供事实依据来减少幻觉 2。进一步减少幻觉的策略包括：
    
    - **强化上下文依赖**：在提示中明确指示LLM严格依据提供的上下文作答，不要依赖其内部知识或进行推测 66。
        
    - **事实核查与验证**：一些高级RAG系统可能包含验证步骤，例如思维链验证（Chain-of-Verification, CoVe），它将问题分解，独立验证子问题的答案，然后综合生成最终答案 65。
        
    - **反馈循环**：建立用户反馈机制，收集关于幻觉的报告，并用于改进检索和生成模块 4。
        
    - **结构化数据检索**：如果适用，从结构化数据源（如知识图谱或数据库）检索信息，可以提供更明确的事实依据 4。
        
    - **选择合适的LLM**：不同LLM在遵循指令和忠实于上下文方面的能力有所不同。某些模型可能更容易产生幻觉。
        
4. 处理长上下文窗口（Handling Long Context Windows） 现代LLM的上下文窗口越来越大（例如，高达100万甚至1000万Token）75。这为RAG带来了新的可能性和挑战：
    
    - **RAG vs. 长上下文直通**：一种观点认为，对于非常长的上下文窗口，可以直接将整个文档（或大量文档）作为上下文传递给LLM，而无需复杂的RAG检索步骤。然而，研究表明，即使上下文窗口很大，LLM也可能无法有效利用所有信息，特别是当关键信息位于上下文中间时（“Lost in the Middle”）75。此外，将大量不相关信息输入LLM会增加计算成本和延迟，并可能引入噪声，反而降低答案质量 18。
        
    - **RAG的持续价值**：尽管上下文窗口增大，RAG在以下方面仍具有优势：
        
        - **选择性与相关性**：RAG能精确检索最相关的片段，避免用大量无关信息淹没LLM 76。
            
        - **处理动态数据**：RAG能从动态更新的外部知识库中检索最新信息，而长上下文直通依赖于输入时提供的数据 76。
            
        - **成本效益**：仅处理相关上下文通常比处理整个长文档更经济 70。
            
        - **可控性与安全性**：RAG允许对检索内容进行更细致的控制，例如基于用户权限过滤信息，这对于企业应用至关重要 73。
            
    - **优化策略**：即使使用长上下文窗口，也可能需要将文档分块，以确保重要信息不被遗漏 75。RAG可以与长上下文窗口结合使用，例如，检索较粗粒度的块或多个相关块，然后将它们组合成一个仍在LLM长上下文窗口范围内的提示。
        

因此，上下文增强是一个涉及提示设计、上下文利用、幻觉控制和适应LLM能力的综合过程。

### **D. 评估**

评估RAG系统的性能对于理解其有效性、发现瓶颈并进行迭代改进至关重要。评估应涵盖检索和生成两个主要阶段。

1. **关键评估指标**
    
    - **检索阶段评估**：
        
        - **上下文精确率（Context Precision）**：衡量检索到的上下文中相关文档（或块）的比例，以及这些相关文档是否排在前面。它关注检索结果的信噪比 79。Ragas框架中的Context Precision评估检索到的上下文中与问题相关的程度 80。
            
        - **上下文召回率（Context Recall）**：衡量所有相关的文档（或块）中有多少被成功检索出来。它关注检索系统找到所有应找到信息的能力 79。
            
        - **上下文相关性（Context Relevance）**：评估检索到的上下文中的句子与问题的相关程度 80。TruLens的RAG三元组评估包含上下文相关性，确保每个上下文块都与输入查询相关 84。DeepEval也包含上下文相关性指标 85。
            
        - **检索排名（Retrieval Ranking）**：评估最相关的文档是否排在检索结果列表的顶部 83。
            
    - **生成阶段评估**：
        
        - **忠实度/真实性（Faithfulness/Groundedness）**：衡量生成的答案是否完全基于提供的上下文，没有引入外部信息或与上下文矛盾。这是评估幻觉的关键指标 79。Ragas、TruLens和DeepEval都包含此类指标 82。
            
        - **答案相关性（Answer Relevance）**：衡量生成的答案是否直接、准确地回应了用户的问题 81。Ragas、TruLens和DeepEval均提供此指标 82。
            
        - **答案正确性（Answer Correctness）**：将生成的答案与“黄金”参考答案进行比较，评估其事实准确性 67。
            
        - **流畅性（Fluency）**：评估生成答案的语言质量，包括语法、清晰度和可读性 79。
            
        - **连贯性（Coherence）**：衡量答案的逻辑流程和思想组织 79。
            
        - **无害性/安全性（Harmlessness/Safety）**：评估答案是否包含偏见、不当言论或有害信息。
            
    - **其他LLM能力评估**：
        
        - **指令遵循能力（Instruction Following）**：评估LLM是否能准确理解并执行提示中的指令 67。
            
        - **处理表格数据能力（Tabular Data QA）**：对于需要处理表格数据的RAG系统，评估模型理解和推理表格内容的能力 86。
            
        - **噪声鲁棒性（Robustness to Noise）**：模型在检索到的文档包含不相关信息时，过滤噪声并关注相关细节的能力 86。
            
        - **反事实鲁棒性（Counterfactual Robustness）**：模型识别和处理检索文档中不正确或误导性信息的能力 86。
            
        - **否定拒绝能力（Negative Rejection）**：当没有足够信息回答查询时，模型能否恰当地拒绝回答 86。
            
2. 评估框架与工具 为了简化和标准化RAG评估过程，出现了一些专门的框架和工具：
    
    - **Ragas**：一个流行的开源框架，用于评估RAG系统。它提供多种指标，如忠实度、答案相关性、上下文精确率和上下文召回率，并可以利用LLM作为“裁判”进行评估 47。Ragas需要问题、理想答案和相关上下文的数据集来进行比较 57。
        
    - **TruLens**：提供“RAG三元组”（上下文相关性、真实性、答案相关性）来评估RAG架构中每个环节的幻觉问题 47。TruLens可以记录和评估LlamaIndex等多模态RAG引擎 88。
        
    - **DeepEval**：另一个用于评估RAG流水线的框架，支持上下文相关性、答案正确性、忠实度等指标。它也使用LLM进行部分指标的计算，因此需要OpenAI API密钥等 47。DeepEval的RAG三元组包括答案相关性、忠实性和上下文相关性 85。
        
    - **Galileo**：提供GenAI Studio等工具，用于测试RAG系统，例如使用10-K年度财务报告进行演示，并计算块归因等指标 9。
        
    - **Arize Phoenix**：一个可视化工具，可以展示RAG流程架构和检索、上下文、生成等步骤的内部情况 57。
        
    - **LangSmith**：LangChain提供的工具，用于调试、测试和监控LangChain应用，包括RAG流水线，并提供评估框架 89。
        
    - **Vertex AI Evaluation Framework**：Google Cloud提供的框架，允许开发者快速实现多种测试指标，并对模型性能进行多次测试 67。
        

建立一个可重复的测试框架至关重要，包括高质量的问题测试集和“黄金”参考输出数据集。在测试时，应遵循一次只改变一个变量的原则，以便准确衡量该变量对系统性能的影响 67。除了自动化指标，人工评估对于判断答案的细微差别（如语气、清晰度）仍然非常重要 67。

## 三、好用的技术

#### 1、MapReduce
可以应用在RAG系统的多个阶段。最常见的是在数据预处理阶段应用，但根据你的具体需求，在检索后或动态选择时应用也是有效的策略。

**数据预处理阶段（索引前）**

```
原始长文档 → MapReduce总结 → 向量化存储
```

工作流程：
1. 获取原始长文档
2. 应用MapReduce总结生成摘要
3. 将摘要而非原文进行分块和向量化
4. 存入向量数据库

优势：
- 减少存储空间和索引时间
- 提前过滤噪音信息
- 加快检索速度

**检索后处理阶段**

```
查询 → 检索原始文档 → MapReduce总结检索结果 → 传递给LLM
```

工作流程：
1. 正常检索原始长文档片段
2. 对检索到的长文档片段应用MapReduce总结
3. 将总结后的内容提供给LLM

优势：
- 保留原始索引的完整性
- 动态根据查询关注点进行总结
- 减少LLM处理的token数量

**混合索引策略**

同时保存原文和摘要：

```
原始长文档 → 分块存储原文 + MapReduce总结 → 双重索引
```

工作流程：
1. 对原始长文档进行常规分块
2. 同时生成整篇文档的MapReduce总结
3. 两种内容都索引到向量数据库中，使用元数据标记

优势：
- 同时具备详细内容和摘要视角
- 可以根据查询复杂度动态选择使用哪种索引
- 在回答时能综合使用不同粒度的信息

## **四、利用开源框架实现RAG**

当前，多个成熟的开源框架为构建RAG系统提供了便利，它们封装了许多底层复杂性，加速了开发进程。本节将概述几个主流框架，并探讨如何利用它们构建RAG应用。

### **A. 主流开源RAG框架概览**

根据GitHub星标数量、社区活跃度以及功能特性，以下是一些广受欢迎的RAG框架 47：

- **LangChain**：一个功能强大的LLM应用开发框架，提供了构建RAG所需的各种组件（数据加载、文本分割、嵌入、向量存储接口、检索器、LLM封装、链、代理等）和丰富的集成。它以其灵活性和组件化设计著称 47。
    
- **LlamaIndex**：专注于将自定义数据源与LLM连接的数据框架，特别擅长数据索引和检索。它提供了灵活的数据连接器、可定制的索引结构和模块化架构，简化了RAG应用的构建 47。
    
- **Haystack (by deepset)**：一个用于构建生产级LLM应用、RAG流水线和先进搜索系统的编排框架。它支持多种文档存储、语言模型，并提供模块化方法进行快速开发 47。
    
- **RAGFlow**：一个基于深度文档理解的开源RAG引擎，特别强调从复杂格式文档中提取结构化信息，并提供可视化界面进行工作流创建 20。
    
- **其他值得关注的框架**：
    
    - **Dify**：一个LLM应用开发平台，支持可视化工作流编辑器，适合非技术用户 47。
        
    - **Embedchain**：简化个性化LLM响应的框架，号称用少于10行代码即可实现 91。
        
    - **txtai**：一个集成的嵌入数据库和LLM编排平台，提供语义搜索、RAG工作流等功能 87。
        
    - **Flowise**：提供拖放式UI来构建定制化LLM流程 47。
        

下表根据 87 的信息，对部分主流框架进行了特性比较：

|框架|主要关注点|最适合场景|关键特性|部署复杂度|GitHub星标 (参考值)|
|---|---|---|---|---|---|
|LangChain|组件链接|通用RAG应用|数据连接、模型灵活性、集成|中等|105k+|
|LlamaIndex|数据索引|自定义知识源|灵活连接器、可定制索引、模块化架构|低|40.8k+|
|Haystack|流水线编排|生产级应用|灵活组件、技术无关、评估工具|中等|20.2k+|
|RAGFlow|文档处理|复杂文档处理|深度文档理解、GraphRAG、可视化界面|中等|48.5k+ 20|
|Dify|可视化开发|非技术用户、企业级|可视化工作流编辑器、广泛模型支持、代理能力|低 (Docker)|90.5k+|
|Milvus|向量存储|大规模向量搜索|先进向量搜索、水平扩展性、混合搜索|中等|33.9k+|
|mem0|持久化记忆|需要上下文保留的助手|多级记忆、自动处理、双重存储|低|27.3k+|
|DSPy|提示优化|需要自我改进的系统|模块化架构、自动提示优化、评估|中等|23k+|
|LightRAG|性能|速度关键型应用|简单架构、信息多样性、全面检索|低|14.6k+|
|LLMWare|资源效率|边缘/CPU部署|高效模型、全面处理、并行化解析|低|12.7k+|
|txtai|一体化解决方案|简化实现|嵌入数据库、流水线组件、多模态支持|低|10.7k+|
|RAGAS|评估|RAG系统测试|客观指标、测试生成、分析仪表盘|低|8.7k+|
|R2R|基于代理的RAG|复杂查询|多模态摄入、代理推理、知识图谱|中等|6.3k+|
|Ragatouille|高级检索|高精度搜索|后期交互检索、微调能力、Token级匹配|中等|3.4k+|
|FlashRAG|研究|实验、基准测试|预处理数据集、算法实现、Web界面|中等|2.1k+|

选择框架时，应考虑易用性、特定功能需求（如复杂文档处理、大规模部署、资源限制）、社区支持和文档完善程度等因素 87。

### **B. LangChain**

LangChain 提供了一套全面的工具和抽象，用于构建端到端的RAG应用。其设计理念是模块化和可组合性，允许开发者灵活地将不同的组件（LLM、嵌入模型、向量存储、检索器等）链接起来形成“链”（Chains）或更复杂的“代理”（Agents）94。

1. **LangChain的RAG架构与关键组件** 10
    
    - **文档加载器 (Document Loaders)**：从多种来源（文件、URL、数据库等）加载数据为Document对象。LangChain支持众多加载器，如WebBaseLoader、PyPDFLoader、CSVLoader等 19。
        
    - **文本分割器 (Text Splitters)**：将加载的Document分割成适合处理的小块。常用分割器有RecursiveCharacterTextSplitter、CharacterTextSplitter等 19。
        
    - **嵌入模型 (Embedding Models)**：将文本块转换为向量嵌入。LangChain集成了OpenAI、Hugging Face、Cohere等多种嵌入模型提供商 94。
        
    - **向量存储 (Vector Stores)**：存储嵌入向量并提供高效的相似性搜索。LangChain支持多种向量数据库，如FAISS、Chroma、Pinecone、Weaviate、PGVector等 46。
        
    - **检索器 (Retrievers)**：根据查询从向量存储中检索相关文档块。LangChain提供多种检索策略，如基本向量存储检索器、多查询检索器、上下文压缩检索器等 41。
        
    - **提示模板 (Prompt Templates)**：用于构建向LLM提问的结构化提示，通常包含用户问题和检索到的上下文。
        
    - **LLM/聊天模型 (LLMs/Chat Models)**：封装了与各种大型语言模型的交互接口。LangChain支持OpenAI、Anthropic、Azure OpenAI、Google Gemini/Vertex AI等众多模型 94。
        
    - **链 (Chains)**：将多个组件按特定顺序组合起来执行复杂任务。RetrievalQA链是RAG的典型实现。
        
    - **LangChain表达式语言 (LCEL)**：一种声明式方式来组合链，使得构建和自定义复杂链更加容易。
        
    - **LangSmith**：用于调试、测试、评估和监控LangChain应用的平台 89。
        
    - **LangGraph**：用于构建更复杂的、有状态的、基于图的代理应用的库，可以实现循环和更灵活的控制流 89。
        
2. **使用LangChain构建简单RAG应用的步骤（基于教程** 19**）** 以下是使用LangChain构建一个基础RAG应用的典型流程：
    
    - **步骤0：环境设置与安装依赖**
        
        - 安装必要的库：langchain, langchain-openai (或其他LLM提供商库), faiss-cpu (或其他向量数据库客户端), beautifulsoup4 (用于网页加载示例), tiktoken (用于Token计算) 等 19。 Bash pip install langchain langchain-openai faiss-cpu beautifulsoup4 tiktoken python-dotenv streamlit # 示例依赖
            
        - 设置API密钥：将OpenAI API密钥等敏感信息存储在环境变量中（例如，通过.env文件和python-dotenv库加载）19。 Python # from dotenv import load_dotenv # load_dotenv() # openai_api_key \= os.getenv("OPENAI_API_KEY")
            
    - **步骤1：加载数据 (Load)**
        
        - 选择并配置一个文档加载器。例如，使用WebBaseLoader加载网页内容 19。 Python # from langchain_community.document_loaders import WebBaseLoader # loader \= WebBaseLoader(web_paths=("[https://lilianweng.github.io/posts/2023-06-23-agent/](https://lilianweng.github.io/posts/2023-06-23-agent/)",)) # docs \= loader.load()
            
    - **步骤2：分割文本 (Split)**
        
        - 使用文本分割器将加载的文档分割成小块。RecursiveCharacterTextSplitter是常用的选择 19。 Python # from langchain_text_splitters import RecursiveCharacterTextSplitter # text_splitter \= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # all_splits \= text_splitter.split_documents(docs)
            
    - **步骤3：创建嵌入并存储 (Store)**
        
        - 初始化一个嵌入模型，例如OpenAIEmbeddings 19。 Python # from langchain_openai import OpenAIEmbeddings # embeddings \= OpenAIEmbeddings() # 默认模型，如 text-embedding-ada-002 或更新型号
            
        - 选择一个向量存储，例如FAISS，并使用分割后的文本块和嵌入模型来构建索引 104。 Python # from langchain_community.vectorstores import FAISS # vector_store \= FAISS.from_documents(all_splits, embeddings)
            
    - **步骤4：创建检索器 (Retrieve)**
        
        - 从向量存储创建一个检索器。检索器负责根据查询提取相关文档块 10。 Python # retriever \= vector_store.as_retriever()
            
    - **步骤5：创建问答链 (Generate/Chain)**
        
        - 初始化一个LLM或聊天模型，例如ChatOpenAI 19。 Python # from langchain_openai import ChatOpenAI # llm \= ChatOpenAI(model_name="gpt-4o-mini", temperature=0) # 使用具体模型名称
            
        - 可以使用LangChain Hub中的预定义RAG提示，或者自定义提示模板 19。 Python # from langchain import hub # rag_prompt \= hub.pull("rlm/rag-prompt")
            
        - 构建一个问答链，例如使用LCEL将检索器、提示和LLM组合起来。 Python # from langchain_core.runnables import RunnablePassthrough # from langchain_core.output_parsers import StrOutputParser # # def format_docs(docs): # return "\n\n".join(doc.page_content for doc in docs) # # rag_chain \= ( # {"context": retriever | format_docs, "question": RunnablePassthrough()} # | rag_prompt # | llm # | StrOutputParser() # ) 或者使用更传统的RetrievalQA链： Python # from langchain.chains import RetrievalQA # qa_chain \= RetrievalQA.from_chain_type( # llm=llm, # chain_type="stuff", # "stuff"是最简单的链类型，将所有检索到的文本直接放入提示中 # retriever=retriever, # return_source_documents=True # 可选，返回源文档 # )
            
    - **步骤6：执行查询并获取结果**
        
        - 调用链的invoke方法（对于LCEL）或直接调用链（对于RetrievalQA）来提问并获取答案 19。 Python # question \= "What is Task Decomposition?" # # For LCEL chain: # response \= rag_chain.invoke(question) # print(response) # # # For RetrievalQA chain: # # result \= qa_chain({"query": question}) # # print(result["result"]) # # if result.get("source_documents"): # # print("Sources:", [doc.metadata.get('source', 'N/A') for doc in result["source_documents"]])
            

LangChain的模块化设计使得替换其中任何组件（如不同的嵌入模型、向量数据库或LLM）都相对容易，同时也支持更高级的RAG技术，如添加聊天历史、上下文压缩、多查询检索等 41。LangGraph进一步增强了构建复杂、有状态的RAG应用的能力 19。

### **C. LlamaIndex**

LlamaIndex 是一个专门为LLM应用设计的数据框架，其核心目标是简化私有或领域特定数据的接入、索引和查询，从而增强LLM的能力 36。它特别强调数据索引的灵活性和查询接口的强大功能。

1. **LlamaIndex的RAG架构与关键组件** 96
    
    - **数据连接器 (Data Connectors / Readers)**：用于从各种来源（API、PDF、SQL、文档等）摄取数据。SimpleDirectoryReader是一个常用的加载器，可以从本地目录加载文件 109。LlamaHub社区提供了大量的数据加载器 96。
        
    - **节点 (Nodes)**：文档被解析和分块后，表示为Node对象，这是LlamaIndex中数据的基本单元，包含文本内容和元数据 108。
        
    - **文本分割器 (Text Splitters / Node Parsers)**：将长文档分割成Node对象。SentenceSplitter是其中一种 108。
        
    - **索引 (Indices)**：LlamaIndex的核心是其多样化的索引结构，用于组织Node对象以便高效查询。最常用的是VectorStoreIndex，它为每个Node生成嵌入向量并存储在向量数据库中 96。其他索引类型包括列表索引、树索引、关键词表索引和知识图谱索引 108。
        
    - **嵌入模型 (Embedding Models)**：用于为文本（节点和查询）生成向量嵌入。LlamaIndex支持OpenAI、Hugging Face等多种嵌入模型，并允许用户自定义 36。
        
    - **向量存储 (Vector Stores)**：与LangChain类似，LlamaIndex也集成了多种向量数据库（如FAISS, Pinecone, Milvus, ChromaDB）用于存储嵌入向量，同时也支持内存存储和本地磁盘持久化 36。
        
    - **检索器 (Retrievers)**：从索引中检索与查询相关的Node对象。可以基于索引类型配置不同的检索模式和参数（如top_k）113。
        
    - **查询引擎 (Query Engines)**：提供了一个高级接口，用于对索引进行自然语言查询并获得LLM生成的响应。RetrieverQueryEngine是常见的查询引擎，它结合了检索器和响应合成模块 106。
        
    - **响应合成器 (Response Synthesizers)**：负责接收检索到的上下文和查询，并使用LLM生成最终答案。LlamaIndex提供了不同的响应模式（如refine, compact, tree_summarize）来控制LLM如何处理上下文并生成响应 107。
        
    - **LLM接口**：封装了与不同LLM（如OpenAI, Replicate上的Llama 2）的交互 36。
        
    - **服务上下文 (ServiceContext) / 设置 (Settings)**：用于配置LlamaIndex应用中的全局设置，如LLM、嵌入模型、分块大小等（在较新版本中，ServiceContext已被Settings全局对象取代）36。
        
    - **LlamaParse**：LlamaIndex提供的先进文档解析服务，特别适用于复杂PDF等文档 107。
        
    - **LlamaCloud**：一个用于数据解析和摄取的托管平台 107。
        
2. **使用LlamaIndex构建简单RAG应用的步骤（基于教程** 109**）** 以下是使用LlamaIndex构建RAG应用的基本流程：
    
    - **步骤0：环境设置与安装依赖**
        
        - 安装核心库：llama-index。根据需要安装LLM提供商（如llama-index-llms-openai）、嵌入模型（如llama-index-embeddings-huggingface）和向量存储（如llama-index-vector-stores-faiss）的集成包 36。 Bash pip install llama-index llama-index-llms-openai llama-index-embeddings-openai faiss-cpu # 示例依赖
            
        - 设置API密钥：将OpenAI API密钥等配置为环境变量 36。 Python # import os # os.environ["OPENAI_API_KEY"] \= "YOUR_OPENAI_API_KEY"
            
    - **步骤1：加载数据 (Load)**
        
        - 使用合适的数据加载器。例如，SimpleDirectoryReader从本地目录加载文档 36。 Python # from llama_index.core import SimpleDirectoryReader # documents \= SimpleDirectoryReader("./data_directory").load_data() # 假设数据在./data_directory 对于PDF，可以使用PyMuPDFReader 111。
            
    - **步骤2：配置LLM和嵌入模型 (可选，但推荐明确指定)**
        
        - 虽然LlamaIndex有默认模型，但通常建议明确配置。 Python # from llama_index.llms.openai import OpenAI # from llama_index.embeddings.openai import OpenAIEmbedding # from llama_index.core import Settings # # Settings.llm \= OpenAI(model="gpt-4o-mini", temperature=0.1) # Settings.embed_model \= OpenAIEmbedding() # 默认使用 text-embedding-ada-002 或更新模型 # # 或者使用HuggingFace模型 # # from llama_index.embeddings.huggingface import HuggingFaceEmbedding # # Settings.embed_model \= HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
    - **步骤3：构建索引 (Index)**
        
        - 使用加载的文档创建索引。VectorStoreIndex是最常用的 36。 Python # from llama_index.core import VectorStoreIndex # index \= VectorStoreIndex.from_documents(documents) 在构建索引时，LlamaIndex会自动进行文本分割（使用默认的SentenceSplitter）和嵌入生成。可以自定义分割器： Python # from llama_index.core.node_parser import SentenceSplitter # from llama_index.core import Settings # Settings.text_splitter \= SentenceSplitter(chunk_size=512, chunk_overlap=50) # index \= VectorStoreIndex.from_documents(documents) # 分割器会在此处应用
            
        - （可选）持久化索引以供后续使用 36： Python # index.storage_context.persist(persist_dir="./storage") # # 后续加载： # # from llama_index.core import StorageContext, load_index_from_storage # # storage_context \= StorageContext.from_defaults(persist_dir="./storage") # # index \= load_index_from_storage(storage_context)
            
    - **步骤4：创建查询引擎 (Query Engine)**
        
        - 从索引创建查询引擎 36。 Python # query_engine \= index.as_query_engine() 可以配置检索参数，如similarity_top_k，或响应模式。
            
    - **步骤5：执行查询并获取响应 (Query)**
        
        - 使用查询引擎的query方法提问 36。 Python # response \= query_engine.query("What did the author do in college?") # print(response) # # 打印源节点 (可选) # # for node in response.source_nodes: # # print(f"Source Node ID: {node.node_id}, Score: {node.score}") # # print(node.get_content()[:200] + "...")
            

LlamaIndex的低级API允许对摄取和检索流程的每个步骤（如手动创建节点、生成嵌入、直接查询向量存储）进行更细致的控制 111。这对于需要高度定制化的应用非常有用。LlamaIndex也强调其可观察性和评估工具的集成，以监控和改进LM应用的性能 95。

### **D. Haystack (by deepset)**

Haystack 是一个开源的LLM编排框架，用于构建可投入生产的、可定制的LLM应用，包括复杂的RAG系统和语义搜索引擎 92。它采用流水线（Pipelines）的方式组织组件，具有技术无关性和灵活性。

1. **Haystack的RAG架构与关键组件** 51
    
    - **文档存储 (Document Stores)**：作为数据的存储后端，Haystack支持多种文档/向量数据库，如Elasticsearch, OpenSearch, FAISS, Milvus, Pinecone, Weaviate, Qdrant, Chroma, AstraDB, MongoDB Atlas, Pgvector等 48。InMemoryDocumentStore适用于小型项目和测试 62。
        
    - **文件转换器 (File Converters)** 和 **预处理器 (Preprocessors)**：用于从不同格式（PDF, DOCX, TXT等）加载文本并进行清洗、分割成Document对象。
        
    - **嵌入器 (Embedders)**：如SentenceTransformersTextEmbedder (用于查询) 和 SentenceTransformersDocumentEmbedder (用于文档)，负责将文本转换为嵌入向量 51。Haystack支持OpenAI、Cohere、Hugging Face等多种嵌入模型。
        
    - **检索器 (Retrievers)**：从文档存储中根据查询检索相关文档。Haystack提供多种类型的检索器：
        
        - **稀疏检索器**：如BM25Retriever (例如 ElasticsearchBM25Retriever, InMemoryBM25Retriever) 51。
            
        - **稠密检索器**：如EmbeddingRetriever (例如 InMemoryEmbeddingRetriever, ElasticsearchEmbeddingRetriever) 51。
            
        - **混合检索器**：结合稀疏和稠密检索。
            
    - **提示构建器 (PromptBuilder / ChatPromptBuilder)**：使用Jinja2模板语言构建发送给LLM的提示，可以动态填充查询和检索到的文档等变量 61。
        
    - **生成器 (Generators / ChatGenerators)**：与LLM交互以生成答案。如OpenAIGenerator或OpenAIChatGenerator 62。
        
    - **流水线 (Pipelines)**：Haystack的核心概念，用于将组件（如检索器、提示构建器、生成器）连接成有向无环图（DAG）来执行复杂任务。Haystack 2.0引入了更动态的流水线，支持控制流和循环 120。
        
    - **代理 (Agents)**：能够使用工具并进行多步推理的更高级抽象（Haystack 2.0中得到增强）。
        
2. **使用Haystack构建简单RAG应用的步骤（基于教程** 62**）** 以下是使用Haystack构建RAG应用的基本流程（主要参考Haystack 2.x版本）：
    
    - **步骤0：环境设置与安装依赖**
        
        - 安装haystack-ai核心包以及特定组件所需的集成包（如elasticsearch-haystack, openai-haystack, sentence-transformers）122。 Bash pip install haystack-ai sentence-transformers openai-haystack # 示例依赖 (InMemoryDocumentStore不需要额外DB依赖)
            
        - 设置API密钥（如OpenAI API密钥）为环境变量 122。 Python # import os # os.environ["OPENAI_API_KEY"] \= "YOUR_OPENAI_API_KEY"
            
    - **步骤1：初始化文档存储 (Initialize Document Store)**
        
        - 选择并初始化一个文档存储。对于简单示例，InMemoryDocumentStore很方便 62。 Python # from haystack.document_stores.in_memory import InMemoryDocumentStore # document_store \= InMemoryDocumentStore() 对于生产环境，可能会选择如ElasticsearchDocumentStore或PineconeDocumentStore等。
            
    - **步骤2：准备并写入文档 (Prepare and Write Documents)**
        
        - 加载数据并转换为Haystack的Document对象。每个Document包含content和可选的meta数据 119。 Python # from haystack.dataclasses import Document # documents \=
            
        - 如果使用稠密检索，需要为文档生成嵌入。初始化一个文档嵌入器，如SentenceTransformersDocumentEmbedder，并运行它来处理文档 62。 Python # from haystack.components.embedders import SentenceTransformersDocumentEmbedder # doc_embedder \= SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2") # doc_embedder.warm_up() # 下载模型 # docs_with_embeddings \= doc_embedder.run(documents=documents)["documents"]
            
        - 将（带有嵌入的）文档写入文档存储 62。 Python # document_store.write_documents(docs_with_embeddings) 或者使用DocumentWriter组件在流水线中写入 122。
            
    - **步骤3：构建RAG流水线 (Build RAG Pipeline)**
        
        - 初始化流水线对象 62。 Python # from haystack import Pipeline # rag_pipeline \= Pipeline()
            
        - **添加组件**：
            
            - **文本嵌入器 (Text Embedder)**：用于嵌入用户查询，应使用与文档嵌入器相同的模型 62。 Python # from haystack.components.embedders import SentenceTransformersTextEmbedder # text_embedder \= SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2") # text_embedder.warm_up() # rag_pipeline.add_component("text_embedder", text_embedder)
                
            - **检索器 (Retriever)**：例如InMemoryEmbeddingRetriever 62。 Python # from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever # retriever \= InMemoryEmbeddingRetriever(document_store=document_store, top_k=3) # top_k指定检索文档数量 # rag_pipeline.add_component("retriever", retriever)
                
            - **提示构建器 (Prompt Builder)**：定义LLM的输入提示模板 62。 Python # from haystack.components.builders.prompt_builder import PromptBuilder # prompt_template \= """ # Given these documents, answer the question. # Documents: # {% for doc in documents %} # {{ doc.content }} # {% endfor %} # Question: {{query}} # Answer: # """ # prompt_builder \= PromptBuilder(template=prompt_template) # rag_pipeline.add_component("prompt_builder", prompt_builder) 对于聊天场景，使用ChatPromptBuilder和ChatMessage 62。
                
            - **LLM生成器 (LLM Generator)**：例如OpenAIChatGenerator 62。 Python # from haystack.components.generators.chat import OpenAIChatGenerator # llm \= OpenAIChatGenerator(model="gpt-4o-mini") # 使用具体模型名称 # rag_pipeline.add_component("llm", llm)
                
        - **连接组件**：定义数据如何在组件间流动 62。 Python # rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding") # rag_pipeline.connect("retriever.documents", "prompt_builder.documents") # rag_pipeline.connect("prompt_builder.prompt", "llm.messages") # 对于ChatGenerator，输入是messages
            
    - **步骤4：运行流水线并获取结果 (Run Pipeline)**
        
        - 使用run()方法执行流水线，并提供必要的输入（如用户查询）62。 Python # query \= "What is Haystack?" # result \= rag_pipeline.run({ # "text_embedder": {"text": query}, # "prompt_builder": {"query": query} # 也需要将原始查询传递给提示构建器 # }) # print(result["llm"]["replies"]) # 假设OpenAIChatGenerator的输出结构 Haystack也提供了预定义的流水线模板，如PredefinedPipeline.RAG或PredefinedPipeline.CHAT_WITH_WEBSITE，可以简化初始设置 120。
            

Haystack的教程和文档提供了更复杂的示例，包括元数据过滤、混合检索、评估等 62。

### **E. RAGFlow**

RAGFlow 是一个基于深度文档理解的开源RAG引擎，其独特之处在于对复杂文档格式（如PDF中的表格、布局、图像）的强大解析能力，以及提供了一个用户友好的Web界面来管理文档和创建RAG工作流 11。

1. **RAGFlow的架构与独特特性** 11
    
    - **深度文档理解 (Deep Document Understanding)**：这是RAGFlow的核心优势。它不仅仅是提取文本，还能理解文档的布局、表格、图片等多模态内容，从而实现更高质量的知识提取 11。2025年3月更新支持使用多模态模型理解PDF/DOCX中的图像 20。
        
    - **模板化分块 (Template-based Chunking)**：RAGFlow提供多种针对不同文档类型（如通用文档、问答对、简历、论文、书籍、法律文件、演示文稿、图片、表格等）的智能分块模板，旨在保证分块的语义完整性和可解释性 14。用户可以为特定文件选择不同于知识库默认设置的分块方法 14。
        
    - **可视化与人工干预**：RAGFlow允许用户可视化文本分块结果，并进行手动干预，如修改文本块内容或为其添加关键词以提高检索排名 14。
        
    - **多源数据兼容**：支持Word、PPT、Excel、PDF、TXT、图片、扫描件、结构化数据、网页等多种复杂格式数据源 11。
        
    - **多路召回与融合重排序 (Multiple Recall paired with Fused Re-ranking)**：采用多种检索策略（如关键词相似度和向量余弦相似度的加权组合）来提高召回率，并结合融合重排序技术优化检索结果的排序 53。支持可选的重排模型（如BCE, BGE reranker），但会增加响应时间 54。
        
    - **有根据的引用 (Grounded Citations)**：生成答案时提供明确的引文来源，快速查看关键参考，减少幻觉 20。
        
    - **知识图谱 (Knowledge Graph)**：支持从文档中提取知识图谱，并在检索中利用图谱进行多跳问答，但这会增加检索时间 20。
        
    - **自动化RAG工作流与API**：提供流线型的RAG编排，支持可配置的LLM和嵌入模型，并提供API以便与业务系统集成 11。
        
    - **用户界面 (Web UI)**：提供直观的Web界面，用于知识库管理、文件上传与解析、分块结果查看、聊天助手创建和Agent工作流构建 20。
        
    - **Agentic RAG**：支持基于图的工作流构建RAG和Agent，允许更复杂的逻辑和工具使用 50。
        
2. **使用RAGFlow构建简单RAG应用的步骤（概念性，基于文档** 14**）** RAGFlow的实现更多依赖其Web UI进行配置和操作，而非纯代码驱动。以下是概念性步骤：
    
    - **步骤0：部署与环境设置**
        
        - 使用Docker部署RAGFlow。官方提供不同版本的Docker镜像（slim版不含嵌入模型，full版包含）20。需要确保系统满足最低配置要求（如CPU、内存、磁盘空间）并正确设置vm.max_map_count 11。
            
        - 配置LLM提供商：在RAGFlow界面中，进入“模型提供商”（Model providers）设置，添加并配置所需的大型语言模型（如DeepSeek-V2, OpenAI GPT系列等）的API密钥 24。RAGFlow支持多种主流LLM 24。
            
        - 配置系统默认模型：设置默认的聊天模型、嵌入模型和图像到文本模型 24。
            
    - **步骤1：创建知识库 (Create Knowledge Base)**
        
        - 在RAGFlow的Web UI中，进入“知识库”选项卡，点击“创建知识库” 24。
            
        - 输入知识库名称。
            
        - **配置知识库** 14：
            
            - **选择分块方法 (Chunking Method)**：从提供的模板中选择最适合文档内容和格式的分块方法（如General, Q&A, Paper, Book等）。
                
            - **选择嵌入模型 (Embedding Model)**：选择用于将文本块转换为向量的嵌入模型。一旦知识库中已有文件被解析，嵌入模型不可更改。
                
    - **步骤2：上传与解析文件 (Upload and Parse Files)**
        
        - 在知识库的“数据集”（Dataset）页面，点击“添加文件”从本地上传文件 24。RAGFlow也支持通过其文件管理系统链接文件到多个知识库 129。
            
        - 对上传的文件点击“解析”（通常是一个播放按钮图标）。文件解析状态变为“SUCCESS”表示完成 24。解析过程包括基于选定模板的分块以及为这些块构建嵌入和全文索引 14。
            
        - （可选）**干预文件解析结果**：点击已解析的文件查看分块结果。可以预览每个块，双击文本块进行修改或添加关键词以提高其检索排名 14。
            
    - **步骤3：运行检索测试 (Run Retrieval Test) (可选但推荐)**
        
        - 在对知识库进行聊天之前，建议运行检索测试以验证分块和检索设置是否能有效召回预期内容 14。
            
        - 可以调整相似度阈值、关键词相似度权重、Top N等参数进行测试 54。
            
    - **步骤4：创建聊天助手或Agent (Create Chat Assistant / Agent)**
        
        - **聊天助手**：
            
            - 进入“聊天”选项卡，点击“创建助手” 24。
                
            - 配置助手名称，并指定要使用的知识库（可多选，但需确保它们使用相同的嵌入模型）24。
                
            - 设置“空回复”：如果希望助手仅基于知识库内容回答，则在此处填写当未检索到答案时的固定回复；若留空，则允许模型在未找到答案时自行发挥（可能产生幻觉）24。
                
            - 配置提示引擎和模型设置（如温度、Top P等）24。
                
        - **Agent**：
            
            - 进入“Agent”选项卡，点击“创建Agent” 128。
                
            - 可以选择从空白模板创建，或使用预设模板（如通用聊天机器人、客服、SQL生成等）。
                
            - 在无代码工作流编辑器中，通过拖放组件（如Begin, Retrieval, Generate, Interact, Message, Keyword, Rewrite等）来构建Agent的逻辑流程 54。
                
            - 配置每个组件的参数，如Retrieval组件的知识库选择、相似度阈值、Top N、是否使用知识图谱等 54；Generate组件的模型选择、自由度（温度/Top P等预设）、系统提示、是否引用等 64。
                
    - **步骤5：与聊天助手或Agent交互**
        
        - 在聊天界面输入问题，RAGFlow将执行配置的RAG流程（检索相关块、增强提示、LLM生成答案并附带引用）并返回结果 24。
            

RAGFlow通过其深度文档理解和用户友好的界面，为处理复杂文档和构建RAG应用提供了一个强大而独特的解决方案。其对多模态内容的处理和知识图谱的集成是其显著特点。

### **F. 框架选择的比较分析与指导**

选择合适的RAG框架取决于项目的具体需求、团队的技术栈、对定制化的要求、以及可用的资源。

- **易用性与快速原型**：
    
    - **LlamaIndex** 和 **LangChain** 提供了大量高层API和预构建组件，上手相对较快，适合快速原型验证和通用RAG应用开发 36。LlamaIndex尤其专注于数据索引和检索的便捷性 95。
        
    - **RAGFlow** 和 **Dify** 提供了可视化界面，降低了非技术人员构建RAG应用的门槛 20。RAGFlow的Web UI使得文档管理、知识库配置和Agent构建更为直观 24。
        
    - **Embedchain** 声称可以用极少的代码实现个性化LLM响应，也适合追求简便性的场景 91。
        
- **复杂文档处理与深度理解**：
    
    - **RAGFlow** 在此方面表现突出，其核心是“深度文档理解”，能够处理PDF、表格、图像等复杂格式，并提取结构化信息 20。其模板化分块和知识图谱功能进一步增强了对复杂文档的处理能力 14。
        
    - **LLMWare** 也专注于高效处理文档，并支持在CPU上运行，适合资源受限的环境 87。
        
- **生产级部署与可扩展性**：
    
    - **Haystack** 被设计为构建生产级应用，其模块化和技术无关性使其能够与多种生产环境组件（如Elasticsearch, OpenSearch等大型文档存储）集成 87。
        
    - **LangChain** 和 **LlamaIndex** 也被广泛用于生产环境，拥有庞大的社区和丰富的集成选项，有助于扩展。
        
    - **Milvus** (作为向量数据库) 和基于它的框架，专注于大规模向量搜索和高吞吐量场景 42。
        
    - RAGFlow也声称其RAG编排适用于个人及大型企业，并支持可配置的LLM和嵌入模型 11。
        
- **定制化与灵活性**：
    
    - **LangChain** 和 **LlamaIndex** 提供了从高层API到低层组件的全面控制，允许开发者根据需求进行深度定制 94。LlamaIndex的低级API尤其适合构建自定义的摄取和检索流程 111。
        
    - **Haystack** 的组件化设计也支持自定义组件的创建和集成 120。
        
    - **DSPy** 框架专注于通过模块化编程和自动提示优化来构建可自我改进的系统，适合需要复杂推理和优化的场景 87。
        
- **特定功能需求**：
    
    - **知识图谱集成**：RAGFlow 20, GraphRAG (Microsoft) 126, R2R 87 等支持知识图谱。
        
    - **Agentic RAG / 多代理系统**：LangChain (LangGraph) 47, LlamaIndex (Agents) 95, RAGFlow (Agent工作流) 50, R2R 87, crewAI 47 等支持构建更复杂的代理系统。
        
    - **持久化记忆**：mem0 专注于为AI应用提供持久化、上下文感知的记忆层 87。
        
    - **评估**：RAGAS 47, TruLens 47, DeepEval 47 是专门的评估框架，一些通用框架如Haystack, LlamaIndex也内置或集成了评估功能。
        
- **资源限制**：
    
    - **LLMWare** 和 **LightRAG** 被认为是资源效率较高，适合在边缘设备或CPU上部署的选项 87。
        
    - RAGFlow提供不同大小的Docker镜像，slim版不含嵌入模型，可以减少资源占用 20。
        
- **社区与生态**：
    
    - **LangChain** 和 **LlamaIndex** 拥有非常庞大和活跃的社区，提供了大量的教程、示例和第三方集成。
        
    - **Haystack** 也有一个成熟的社区和丰富的集成。
        

最终，没有“最好”的框架，只有“最适合”的框架。建议开发者根据项目需求，从小处着手，尝试几个候选框架，评估其是否满足关键需求，然后再做出最终选择。对于初学者，从提供高层API和清晰教程的框架（如LangChain, LlamaIndex）入手可能更容易。对于需要处理大量复杂文档并希望有UI支持的用户，RAGFlow值得考虑。对于需要构建高度定制化或研究性系统的开发者，LangChain, LlamaIndex, DSPy等提供了更大的灵活性。

## **五、从零开始构建RAG系统（概念性）**

虽然开源框架极大地简化了RAG系统的构建，但从概念上理解如何“从零开始”搭建一个RAG系统，有助于深化对RAG核心机制的认识，并为将来进行更深层次的定制或问题排查打下基础。正如LlamaIndex的倡导者所言，从头构建有助于真正理解各个组件 130。

### **A. 构建RAG的基本原理与学习价值**

从零开始构建RAG，意味着不依赖于现成的RAG框架（如LangChain或LlamaIndex），而是直接使用基础库（如用于HTTP请求的requests，用于向量计算的numpy或scikit-learn，以及LLM和嵌入模型的API客户端）来实现RAG的核心流程：数据加载、分块、嵌入、存储、检索、提示构建和LLM调用 130。

**学习价值**：

1. **深入理解核心机制**：手动实现每个步骤能清晰地揭示数据如何在RAG管道中流动，各个组件（如嵌入模型、向量相似度计算、提示模板）的具体作用和相互关系。
    
2. **掌握底层技术细节**：了解如何处理API调用、数据格式转换、相似度计算等底层细节，这对于优化性能和解决复杂问题至关重要。
    
3. **增强定制能力**：一旦理解了基本原理，就能更容易地对现有框架进行定制，或者在没有合适框架组件时自行实现特定功能。
    
4. **更好地评估框架**：理解了“幕后”发生的事情后，能更明智地选择和使用RAG框架，理解其抽象的真正含义。
    

### **B. 核心组件的简化实现思路**

以下是一个极简RAG流程中各核心组件的实现思路，主要参考 130 和 130 中的概念：

1. **数据准备 (Data Preparation)**
    
    - **文档语料库 (Corpus of Documents)**：首先需要一个文档集合。在最简单的情况下，这可以是一个字符串列表，每个字符串代表一个文档或一个预先分好的块。 Python # corpus_of_documents \= ["文档1的内容...", "文档2的内容...",...]
        
    - **分块 (Chunking)**：如果文档较长，需要手动将其分割成小块。可以使用简单的基于字符数、句子或段落的分割逻辑。 Python # def simple_chunker(text, chunk_size, overlap):... # chunks \= # for doc_content in raw_documents: # chunks.extend(simple_chunker(doc_content, 500, 50))
        
    - **嵌入 (Embedding)**：选择一个嵌入模型（例如，通过Sentence Transformers库直接调用开源模型，或通过API调用商业模型如OpenAI）。为每个文本块和用户查询生成嵌入向量。 Python # from sentence_transformers import SentenceTransformer # embed_model \= SentenceTransformer('all-MiniLM-L6-v2') # 示例模型 # # document_embeddings \= embed_model.encode(chunks) # query_embedding \= embed_model.encode(user_query)
        
    - **存储 (Storing)**：对于小型实验，可以将文本块及其嵌入向量存储在Python字典、列表或简单的本地文件（如JSON或CSV）中。对于稍大规模，可以使用SQLite。更专业的做法是使用向量数据库客户端。 Python # # 简单存储示例 # indexed_data \= # for i, chunk_text in enumerate(chunks): # indexed_data.append({"text": chunk_text, "embedding": document_embeddings[i]})
        
2. **检索 (Retrieval)**
    
    - **相似度计算**：计算用户查询嵌入与所有文档块嵌入之间的相似度。余弦相似度是常用的度量。 Python # from sklearn.metrics.pairwise import cosine_similarity # import numpy as np # # similarities \= cosine_similarity(query_embedding.reshape(1, -1), np.array(document_embeddings)) # # similarities 是一个包含查询与所有文档块相似度的数组 130 中使用Jaccard相似度作为更简单的示例（非嵌入基础），说明了相似度比较的概念。
        
    - **选择Top-K块**：根据相似度得分，选择最相似的K个文本块作为上下文。 Python # top_k_indices \= np.argsort(similarities)[::-1][:k] # k是希望检索的块数量 # retrieved_context_chunks \= [indexed_data[i]["text"] for i in top_k_indices]
        
3. **生成 (Generation)**
    
    - **提示构建 (Prompt Construction)**：将用户查询和检索到的上下文块组合成一个发送给LLM的提示。 Python # context_string \= "\n\n".join(retrieved_context_chunks) # prompt \= f"""基于以下上下文信息回答问题： # 上下文： # {context_string} # # 问题：{user_query} # 回答：""" 130 提供了一个更具体的提示模板示例。
        
    - **LLM调用 (LLM Invocation)**：使用requests库或特定LLM提供商的Python客户端库（如openai）将提示发送给LLM，并获取生成的响应。 Python # # 使用OpenAI API的简化示例 # # import openai # # client \= openai.OpenAI(api_key="YOUR_API_KEY") # # response \= client.chat.completions.create( # # model="gpt-4o-mini", # 或其他模型 # # messages=[{"role": "user", "content": prompt}] # # ) # # answer \= response.choices.message.content # # # [130] 中使用requests调用本地Ollama服务的示例： # # import requests # # import json # # url \= '[http://localhost:11434/api/generate](http://localhost:11434/api/generate)' # # data \= { "model": "llama2", "prompt": prompt_filled_template } # # headers \= {'Content-Type': 'application/json'} # # response \= requests.post(url, data=json.dumps(data), headers=headers, stream=True) # # # 处理流式响应...
        
    - **解析响应 (Parsing Response)**：从LLM的API响应中提取生成的文本。
        

### **C. 组件集成的考量因素**

在从零开始集成这些组件时，需要考虑：

- **数据流管理**：确保数据在各个步骤之间正确传递和转换。
    
- **错误处理**：为API调用失败、数据格式错误等情况添加健壮的错误处理逻辑。
    
- **配置管理**：将模型名称、API密钥、top_k值等参数外部化为配置，而不是硬编码。
    
- **性能**：对于大量数据，简单的Python列表和循环进行相似度搜索会非常慢。这时就需要理解向量数据库为何如此重要（它们使用优化的索引结构如HNSW进行ANN近邻搜索）。
    
- **可扩展性**：从零开始的简单实现可能难以扩展到处理大量数据或高并发请求。
    

通过手动实现这些步骤，开发者能更深刻地体会到RAG框架所提供的抽象和便利性，并为未来构建更复杂、更优化的RAG系统打下坚实基础。这个过程也揭示了为何选择合适的嵌入模型、分块策略和向量数据库对RAG性能至关重要。

## **六、高级主题与最佳实践**

构建一个基础的RAG系统后，通常需要进一步优化和扩展其功能，以应对更复杂的应用场景和性能要求。

### **A. 处理动态与实时数据**

许多RAG应用需要处理不断变化的数据源，确保LLM能够访问最新的信息。

1. 嵌入的更新策略 当外部知识库中的数据发生变化（新增、修改、删除）时，相应的嵌入向量也需要更新，以保持检索的准确性 2。
    
    - **定期完全重嵌入 (Periodic Full Re-embedding)**：定期（如每日、每周）对整个数据集或发生变化的部分重新生成嵌入并更新向量数据库。这适用于数据变化不是非常频繁的场景 39。
        
    - **增量更新 (Incremental Updates)**：仅对新增或修改的文档块生成嵌入并添加到向量数据库中，或更新现有嵌入。对于删除的文档块，则从向量数据库中移除其嵌入。这种方式计算成本较低，适合数据频繁更新的场景 12。
        
    - **触发式更新 (Trigger-based Updates)**：当检测到源数据发生变化时，自动触发受影响部分的重索引过程 12。例如，事件驱动架构（如使用Apache Kafka）可以将事务性更新作为实时事件传播，确保RAG系统访问最新数据 40。
        
    - **混合方法**：结合静态嵌入（用于不常变化的基础知识）和动态更新的表示（如为嵌入附加时间戳等元数据），使检索系统能够优先考虑最新文档，而无需完全改变核心向量 39。
        
    - **版本控制**：对文档语料库和嵌入进行版本控制（快照），有助于跟踪变化、回滚以及评估不同版本嵌入对性能的影响 12。
        
2. 实时数据流集成 对于需要即时信息的应用（如新闻摘要、社交媒体监控），RAG系统可以直接连接到实时数据流。这要求数据摄取和嵌入更新流程具有低延迟特性。 更新嵌入会直接影响RAG评估。嵌入的变化可能改变查询与文档间的相似度得分，从而影响检索内容。因此，在更新嵌入后，必须重新评估检索指标（如精确率、召回率），以确保系统性能稳定 39。
    

### **B. 优化成本、延迟与可扩展性**

1. **成本优化**
    
    - **选择合适的LLM和嵌入模型**：商业模型的API调用会产生费用。开源模型虽然本身免费，但部署和维护需要计算资源。应根据性能需求和预算权衡 27。较小或量化后的模型可以降低计算成本 131。
        
    - **嵌入维度**：较低维度的嵌入向量占用存储空间更少，查询更快，但可能损失部分语义信息。一些模型支持可变维度嵌入（如OpenAI text-embedding-3系列），允许在性能和成本间进行权衡 23。
        
    - **高效分块与去重**：优化分块策略，避免产生过多冗余或过小的块。对数据进行去重，减少不必要的嵌入生成和存储 13。
        
    - **缓存**：缓存频繁访问的嵌入、检索结果或LLM生成结果，可以减少重复计算和API调用 78。
        
    - **批量处理**：在嵌入生成和数据索引时使用批量处理，可以提高效率 37。
        
2. **延迟优化**
    
    - **高效的向量数据库和索引**：选择针对低延迟查询优化的向量数据库，并配置合适的索引参数（如HNSW的构建参数）42。
        
    - **硬件加速**：利用GPU或专用AI硬件（如FPGA、Google TPU）加速嵌入生成和向量搜索 37。
        
    - **近似最近邻搜索 (ANN)**：ANN算法（如HNSW, IVF）通过牺牲一定的召回精度来换取查询速度的大幅提升，适用于大规模数据集 37。
        
    - **预取与并行处理**：例如，TeleRAG系统提出的“前瞻性检索”（lookahead retrieval）机制，在LLM生成（预检索生成阶段）的同时，预测并预先将可能需要的IVF集群数据从CPU传输到GPU，从而隐藏数据传输开销 37。
        
    - **轻量级重排模型**：如果使用重排器，选择计算效率较高的模型，或者只对少量候选结果进行重排。
        
    - **流式输出 (Streaming)**：对于LLM的生成步骤，使用流式输出可以将答案逐步展示给用户，改善感知延迟 10。RAGFlow v0.7.0开始支持流式输出 50。
        
3. **可扩展性设计**
    
    - **分布式架构**：对于大规模应用，应采用分布式架构部署RAG系统的各个组件（如向量数据库集群、可水平扩展的LLM推理服务）42。Kubernetes是常用的部署和编排工具 112。
        
    - **数据分区/分片 (Data Partitioning/Sharding)**：将大规模向量数据分割成多个分片，分布在不同的节点上，以实现水平扩展 42。
        
    - **异步处理与消息队列**：对于非实时或耗时较长的任务（如批量索引更新），使用异步处理和消息队列（如Redis, RabbitMQ）可以提高系统的吞吐量和响应能力 112。
        
    - **负载均衡**：在服务入口处部署负载均衡器，将请求分发到后端的多个RAG实例。
        

### **C. 安全与伦理考量**

1. **数据隐私与访问控制**
    
    - RAG系统常用于处理企业内部的敏感或私有数据。必须确保在数据加载、存储、检索和生成过程中，严格遵守数据隐私法规（如GDPR）并实施适当的访问控制 6。
        
    - 应能根据用户角色和权限过滤检索结果，确保用户只能访问其有权查看的信息 73。
        
    - 对敏感数据（如PII）在摄入时进行脱敏或屏蔽处理 132。
        
2. **偏见与公平性**
    
    - LLM和嵌入模型可能继承训练数据中的偏见。RAG系统所依赖的外部知识库也可能包含偏见信息。需要警惕这些偏见，并努力减轻其在生成结果中的影响。
        
    - 定期审计数据源和模型输出，以识别和纠正潜在的偏见问题。
        
3. **内容安全与护栏**
    
    - 实施内容安全护栏，防止RAG系统生成有害、不当或攻击性的内容 5。
        
    - 对于用户输入也应进行校验，防止恶意提示注入（Prompt Injection）等攻击。
        
4. **透明度与可解释性**
    
    - 提供答案来源的引用是提高RAG系统透明度和用户信任度的重要手段 5。用户应能追溯信息来源，验证答案的准确性。
        
    - 记录RAG流程中各步骤的输入输出，有助于诊断问题和理解系统行为 12。
        

通过综合考虑这些高级主题和最佳实践，可以构建出更强大、高效、可靠且负责任的RAG系统。

## **七、结论与未来展望**

检索增强生成（RAG）技术通过将大型语言模型与外部知识库相结合，显著提升了AI系统在处理知识密集型任务时的准确性、相关性和时效性。本报告对RAG的核心概念、关键组件、构建流程、主流开源框架（LangChain, LlamaIndex, Haystack, RAGFlow）及其实现方法进行了深入探讨，并分析了数据准备、检索机制、上下文增强、评估方法等关键环节。

分析表明，成功的RAG系统构建依赖于对数据处理流程的精心设计——从智能分块策略（如内容感知分块、模板化分块）和高质量元数据利用，到选择合适的嵌入模型和向量数据库。高级检索技术如混合检索、重排序和知识图谱集成，能够进一步优化检索结果的相关性。在生成阶段，有效的提示工程、上下文管理以及减少幻觉的策略至关重要。此外，全面的评估框架和指标是持续改进RAG系统性能的保障。

开源框架为RAG的实现提供了强大支持。LangChain和LlamaIndex以其灵活性和丰富的组件生态系统，成为通用RAG开发的热门选择。Haystack则更侧重于生产级应用的编排。RAGFlow凭借其深度文档理解能力和用户友好的界面，在处理复杂文档方面展现出独特优势。选择何种框架，或是否从零开始构建，取决于具体项目需求、团队技能和对系统控制的程度。

展望未来，RAG技术的发展将呈现以下趋势：

1. **更智能的检索与融合**：随着模型理解能力的增强，检索机制将更加智能化，能够更好地理解用户意图和上下文细微差别。多模态检索、知识图谱与向量检索的深度融合，以及更先进的重排序和结果融合技术将持续涌现。
    
2. **Agentic RAG的兴起**：将RAG与AI代理（Agent）结合，赋予系统更强的自主规划、工具使用和多步推理能力，使其能够处理更复杂的任务，动态调整检索和生成策略 21。
    
3. **长上下文LLM与RAG的协同进化**：尽管LLM的上下文窗口不断扩大，但RAG在处理海量、动态变化的知识以及提供可追溯性方面的优势仍将持续。未来，两者将更紧密地协同工作，例如RAG负责初步筛选和浓缩信息，再由长上下文LLM进行深度理解和生成。
    
4. **自动化与自适应RAG**：通过AutoML等技术实现RAG流水线中超参数（如分块大小、嵌入模型、提示模板）的自动优化。系统将能根据数据特征和查询类型自适应地调整其策略。
    
5. **端到端优化与评估的深化**：开发更全面的、能够评估整个RAG流水线端到端性能的指标和工具，并关注用户满意度等更贴近实际应用的评估维度。
    
6. **多模态RAG的普及**：RAG系统将不仅处理文本，还将扩展到图像、音频、视频等多种数据模态，实现真正的多模态信息检索与生成。
    

总之，RAG作为连接LLM与海量知识的关键桥梁，其重要性日益凸显。通过不断的技术创新和最佳实践积累，RAG系统将在更广泛的领域释放AI的潜力，提供更智能、更可靠、更值得信赖的信息服务。

#### **Works cited**

1. arxiv.org, accessed May 10, 2025, [https://arxiv.org/abs/2503.18016#:~:text=Retrieval%2Daugmented%20generation%20(RAG),%2Dto%2Ddate%20knowledge%20sources.](https://arxiv.org/abs/2503.18016#:~:text=Retrieval%2Daugmented%20generation%20\\(RAG\\),%2Dto%2Ddate%20knowledge%20sources.)
    
2. What is RAG? - Retrieval-Augmented Generation AI Explained - AWS, accessed May 10, 2025, [https://aws.amazon.com/what-is/retrieval-augmented-generation/](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
    
3. Retrieval Augmented Generation (RAG) for LLMs - Prompt Engineering Guide, accessed May 10, 2025, [https://www.promptingguide.ai/research/rag](https://www.promptingguide.ai/research/rag)
    
4. How to Prevent AI Hallucinations with Retrieval Augmented Generation - IT Convergence, accessed May 10, 2025, [https://www.itconvergence.com/blog/how-to-overcome-ai-hallucinations-using-retrieval-augmented-generation/](https://www.itconvergence.com/blog/how-to-overcome-ai-hallucinations-using-retrieval-augmented-generation/)
    
5. Understanding Retrieval Augmented Generation - AWS Prescriptive Guidance, accessed May 10, 2025, [https://docs.aws.amazon.com/prescriptive-guidance/latest/retrieval-augmented-generation-options/what-is-rag.html](https://docs.aws.amazon.com/prescriptive-guidance/latest/retrieval-augmented-generation-options/what-is-rag.html)
    
6. Augment LLMs with RAGs or Fine-Tuning - Learn Microsoft, accessed May 10, 2025, [https://learn.microsoft.com/en-us/azure/developer/ai/augment-llm-rag-fine-tuning](https://learn.microsoft.com/en-us/azure/developer/ai/augment-llm-rag-fine-tuning)
    
7. What is Retrieval-Augmented Generation (RAG)? A Practical Guide - K2view, accessed May 10, 2025, [https://www.k2view.com/what-is-retrieval-augmented-generation](https://www.k2view.com/what-is-retrieval-augmented-generation)
    
8. Mastering RAG: A Deep Dive into Retrieval Augmented Generation - Valprovia, accessed May 10, 2025, [https://www.valprovia.com/en/blog/mastering-rag-a-deep-dive-into-retrieval-augmented-generation](https://www.valprovia.com/en/blog/mastering-rag-a-deep-dive-into-retrieval-augmented-generation)
    
9. Explaining RAG Architecture: A Deep Dive into Components | Galileo.ai, accessed May 10, 2025, [https://www.galileo.ai/blog/rag-architecture](https://www.galileo.ai/blog/rag-architecture)
    
10. Retrieval augmented generation (RAG) - ️ LangChain, accessed May 10, 2025, [https://python.langchain.com/docs/concepts/rag/](https://python.langchain.com/docs/concepts/rag/)
    
11. RAGFlow: An Open Source RAG Engine Based on Deep Document Understanding to Provide Efficient Retrieval Enhanced Generation Workflow - 首席AI分享圈, accessed May 10, 2025, [https://www.aisharenet.com/en/ragflow/](https://www.aisharenet.com/en/ragflow/)
    
12. Build Advanced Retrieval-Augmented Generation Systems | Microsoft Learn, accessed May 10, 2025, [https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation](https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation)
    
13. Build an unstructured data pipeline for RAG - Databricks Documentation, accessed May 10, 2025, [https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)
    
14. Configure knowledge base - RAGFlow, accessed May 10, 2025, [https://ragflow.io/docs/dev/configure_knowledge_base](https://ragflow.io/docs/dev/configure_knowledge_base)
    
15. Build an unstructured data pipeline for RAG - Azure Databricks | Microsoft Learn, accessed May 10, 2025, [https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)
    
16. Chunking strategies for RAG tutorial using Granite - IBM, accessed May 10, 2025, [https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai](https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai)
    
17. 7 Chunking Strategies in RAG You Need To Know - F22 Labs, accessed May 10, 2025, [https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/](https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/)
    
18. Long-Context Isn't All You Need: How Retrieval & Chunking Impact Finance RAG, accessed May 10, 2025, [https://www.snowflake.com/en/engineering-blog/impact-retrieval-chunking-finance-rag/](https://www.snowflake.com/en/engineering-blog/impact-retrieval-chunking-finance-rag/)
    
19. Build a Retrieval Augmented Generation (RAG) App: Part 1 ..., accessed May 10, 2025, [https://js.langchain.com/docs/tutorials/rag/](https://js.langchain.com/docs/tutorials/rag/)
    
20. infiniflow/ragflow: RAGFlow is an open-source RAG ... - GitHub, accessed May 10, 2025, [https://github.com/infiniflow/ragflow](https://github.com/infiniflow/ragflow)
    
21. [2501.09136] Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG - arXiv, accessed May 10, 2025, [https://arxiv.org/abs/2501.09136](https://arxiv.org/abs/2501.09136)
    
22. Leveraging Fine-Tuned Retrieval-Augmented Generation with Long-Context Support: For 3GPP Standards - arXiv, accessed May 10, 2025, [https://arxiv.org/html/2408.11775v2](https://arxiv.org/html/2408.11775v2)
    
23. 6 Data Processing Steps for RAG: Precision and Performance - Galileo AI, accessed May 10, 2025, [https://www.galileo.ai/blog/data-processing-steps-rag-precision-performance](https://www.galileo.ai/blog/data-processing-steps-rag-precision-performance)
    
24. Get started - RAGFlow, accessed May 10, 2025, [https://ragflow.io/docs/dev/](https://ragflow.io/docs/dev/)
    
25. Best practices for structuring large datasets in Retrieval-Augmented Generation (RAG) - DataScienceCentral.com, accessed May 10, 2025, [https://www.datasciencecentral.com/best-practices-for-structuring-large-datasets-in-retrieval-augmented-generation-rag/](https://www.datasciencecentral.com/best-practices-for-structuring-large-datasets-in-retrieval-augmented-generation-rag/)
    
26. Using Metadata in Retrieval-Augmented Generation - Deasy Labs: Efficient Metadata Solutions for Scalable AI Workflows, accessed May 10, 2025, [https://www.deasylabs.com/blog/using-metadata-in-retrieval-augmented-generation](https://www.deasylabs.com/blog/using-metadata-in-retrieval-augmented-generation)
    
27. How to Choose the Right Embedding for Your RAG Model? - Analytics Vidhya, accessed May 10, 2025, [https://www.analyticsvidhya.com/blog/2025/03/embedding-for-rag-models/](https://www.analyticsvidhya.com/blog/2025/03/embedding-for-rag-models/)
    
28. Embeddings - IBM, accessed May 10, 2025, [https://www.ibm.com/architectures/papers/rag-cookbook/embedding](https://www.ibm.com/architectures/papers/rag-cookbook/embedding)
    
29. Mastering RAG: How to Select an Embedding Model - Galileo AI, accessed May 10, 2025, [https://www.galileo.ai/blog/mastering-rag-how-to-select-an-embedding-model](https://www.galileo.ai/blog/mastering-rag-how-to-select-an-embedding-model)
    
30. Choose the best embedding model for your Retrieval-augmented generation (RAG) system, accessed May 10, 2025, [https://www.enterprisebot.ai/blog/choose-the-best-embedding-model-for-your-retrieval-augmented-generation-rag-system](https://www.enterprisebot.ai/blog/choose-the-best-embedding-model-for-your-retrieval-augmented-generation-rag-system)
    
31. How to Fine-Tune Embedding Models for RAG Systems - Dataworkz - RAG as a Service, accessed May 10, 2025, [https://www.dataworkz.com/blog/how-to-fine-tune-embedding-models-for-rag-systems/](https://www.dataworkz.com/blog/how-to-fine-tune-embedding-models-for-rag-systems/)
    
32. Local LLMs vs. OpenAI for RAG: Accuracy & Cost Comparison - Chitika, accessed May 10, 2025, [https://www.chitika.com/local-llm-vs-openai-rag/](https://www.chitika.com/local-llm-vs-openai-rag/)
    
33. Evaluating Open-Source vs. OpenAI Embeddings for RAG: A How-To Guide - Timescale, accessed May 10, 2025, [https://www.timescale.com/blog/open-source-vs-openai-embeddings-for-rag](https://www.timescale.com/blog/open-source-vs-openai-embeddings-for-rag)
    
34. Choosing the Best Embedding Models for RAG and Document Understanding - Beam Cloud, accessed May 10, 2025, [https://www.beam.cloud/blog/best-embedding-models](https://www.beam.cloud/blog/best-embedding-models)
    
35. How to Choose the Best Embedding Model for Your LLM Application | MongoDB, accessed May 10, 2025, [https://www.mongodb.com/developer/products/atlas/choose-embedding-model-rag/](https://www.mongodb.com/developer/products/atlas/choose-embedding-model-rag/)
    
36. LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - GitHub, accessed May 10, 2025, [https://github.com/markmcd/llamaindex](https://github.com/markmcd/llamaindex)
    
37. TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval, accessed May 10, 2025, [https://arxiv.org/html/2502.20969v1](https://arxiv.org/html/2502.20969v1)
    
38. Accelerating Retrieval-Augmented Generation - ARG | ECE at Cornell, accessed May 10, 2025, [https://arg.csl.cornell.edu/papers/iks-asplos2025.pdf](https://arg.csl.cornell.edu/papers/iks-asplos2025.pdf)
    
39. What strategies can be used to update or improve embeddings over time as new data becomes available, and how would that affect ongoing RAG evaluations? - Milvus, accessed May 10, 2025, [https://milvus.io/ai-quick-reference/what-strategies-can-be-used-to-update-or-improve-embeddings-over-time-as-new-data-becomes-available-and-how-would-that-affect-ongoing-rag-evaluations](https://milvus.io/ai-quick-reference/what-strategies-can-be-used-to-update-or-improve-embeddings-over-time-as-new-data-becomes-available-and-how-would-that-affect-ongoing-rag-evaluations)
    
40. Integrating Transaction Processing with RAG Systems - Chitika, accessed May 10, 2025, [https://www.chitika.com/integrating-rag-transaction-processing/](https://www.chitika.com/integrating-rag-transaction-processing/)
    
41. How-to guides | 🦜️ LangChain, accessed May 10, 2025, [https://python.langchain.com/v0.2/docs/how_to/#retrieval-augmented-generation-rag](https://python.langchain.com/v0.2/docs/how_to/#retrieval-augmented-generation-rag)
    
42. Architecting for Scale: Evaluating Vector Database Options for Production RAG Systems, accessed May 10, 2025, [https://ragaboutit.com/architecting-for-scale-evaluating-vector-database-options-for-production-rag-systems/](https://ragaboutit.com/architecting-for-scale-evaluating-vector-database-options-for-production-rag-systems/)
    
43. Best Vector Database for RAG: Qdrant vs. Weaviate vs. Pinecone - Research AIMultiple, accessed May 10, 2025, [https://research.aimultiple.com/vector-database-for-rag/](https://research.aimultiple.com/vector-database-for-rag/)
    
44. Vector database choices in Vertex AI RAG Engine - Google Cloud, accessed May 10, 2025, [https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/vector-db-choices?hl=en](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/vector-db-choices?hl=en)
    
45. Vector database for RAG applications - Aerospike, accessed May 10, 2025, [https://aerospike.com/solutions/use-cases/rag-vector-database/](https://aerospike.com/solutions/use-cases/rag-vector-database/)
    
46. LangChain - Mem0 docs, accessed May 10, 2025, [https://docs.mem0.ai/components/vectordbs/dbs/langchain](https://docs.mem0.ai/components/vectordbs/dbs/langchain)
    
47. Awesome-RAG: a curated list of Retrieval-Augmented Generation - GitHub, accessed May 10, 2025, [https://github.com/noworneverev/Awesome-RAG](https://github.com/noworneverev/Awesome-RAG)
    
48. Choosing a Document Store - Haystack Documentation - Deepset, accessed May 10, 2025, [https://docs.haystack.deepset.ai/docs/choosing-a-document-store](https://docs.haystack.deepset.ai/docs/choosing-a-document-store)
    
49. Vector stores - ️ LangChain, accessed May 10, 2025, [https://python.langchain.com/v0.1/docs/integrations/vectorstores/](https://python.langchain.com/v0.1/docs/integrations/vectorstores/)
    
50. Release notes | RAGFlow, accessed May 10, 2025, [https://ragflow.io/docs/v0.16.0/release_notes](https://ragflow.io/docs/v0.16.0/release_notes)
    
51. Retrievers - Haystack Documentation - Deepset, accessed May 10, 2025, [https://docs.haystack.deepset.ai/docs/retrievers](https://docs.haystack.deepset.ai/docs/retrievers)
    
52. davidsbatista/haystack-retrieval: Different retrieval techniques implemented in Haystack - GitHub, accessed May 10, 2025, [https://github.com/davidsbatista/haystack-retrieval](https://github.com/davidsbatista/haystack-retrieval)
    
53. ragflow.io, accessed May 10, 2025, [https://ragflow.io/docs/dev/retrieval_component#:~:text=RAGFlow%20employs%20a%20combination%20of,be%20excluded%20from%20the%20results.](https://ragflow.io/docs/dev/retrieval_component#:~:text=RAGFlow%20employs%20a%20combination%20of,be%20excluded%20from%20the%20results.)
    
54. Retrieval component | RAGFlow, accessed May 10, 2025, [https://ragflow.io/docs/dev/retrieval_component](https://ragflow.io/docs/dev/retrieval_component)
    
55. Implementation of all RAG techniques in a simpler way - GitHub, accessed May 10, 2025, [https://github.com/FareedKhan-dev/all-rag-techniques](https://github.com/FareedKhan-dev/all-rag-techniques)
    
56. NLP • Retrieval Augmented Generation - aman.ai, accessed May 10, 2025, [https://aman.ai/primers/ai/RAG/](https://aman.ai/primers/ai/RAG/)
    
57. Best Practices in RAG Evaluation: A Comprehensive Guide - Qdrant, accessed May 10, 2025, [https://qdrant.tech/blog/rag-evaluation-guide/](https://qdrant.tech/blog/rag-evaluation-guide/)
    
58. Re-ranking in Retrieval Augmented Generation: How to Use Re-rankers in RAG - Chitika, accessed May 10, 2025, [https://www.chitika.com/re-ranking-in-retrieval-augmented-generation-how-to-use-re-rankers-in-rag/](https://www.chitika.com/re-ranking-in-retrieval-augmented-generation-how-to-use-re-rankers-in-rag/)
    
59. RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. - GitHub, accessed May 10, 2025, [https://github.com/nexussdad/ragflow-kjt](https://github.com/nexussdad/ragflow-kjt)
    
60. RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. - GitHub, accessed May 10, 2025, [https://github.com/StickPromise/RAGflow](https://github.com/StickPromise/RAGflow)
    
61. PromptBuilder - Haystack Documentation - Deepset, accessed May 10, 2025, [https://docs.haystack.deepset.ai/docs/promptbuilder](https://docs.haystack.deepset.ai/docs/promptbuilder)
    
62. Creating Your First QA Pipeline with Retrieval-Augmentation ..., accessed May 10, 2025, [https://haystack.deepset.ai/tutorials/27_first_rag_pipeline](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline)
    
63. PromptBuilder - Haystack Documentation - Deepset, accessed May 10, 2025, [https://docs.haystack.deepset.ai/v2.4/docs/promptbuilder](https://docs.haystack.deepset.ai/v2.4/docs/promptbuilder)
    
64. Generate component | RAGFlow, accessed May 10, 2025, [https://ragflow.io/docs/dev/generate_component](https://ragflow.io/docs/dev/generate_component)
    
65. RAG LLM Prompting Techniques to Reduce Hallucinations - Galileo AI, accessed May 10, 2025, [https://www.galileo.ai/blog/mastering-rag-llm-prompting-techniques-for-reducing-hallucinations](https://www.galileo.ai/blog/mastering-rag-llm-prompting-techniques-for-reducing-hallucinations)
    
66. 3 Recommended Strategies to Reduce LLM Hallucinations - Vellum AI, accessed May 10, 2025, [https://www.vellum.ai/blog/how-to-reduce-llm-hallucinations](https://www.vellum.ai/blog/how-to-reduce-llm-hallucinations)
    
67. RAG systems: Best practices to master evaluation for accurate and reliable AI. | Google Cloud Blog, accessed May 10, 2025, [https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval)
    
68. The Needle In a Haystack Test: Evaluating the Performance of LLM RAG Systems - Arize AI, accessed May 10, 2025, [https://arize.com/blog-course/the-needle-in-a-haystack-test-evaluating-the-performance-of-llm-rag-systems/](https://arize.com/blog-course/the-needle-in-a-haystack-test-evaluating-the-performance-of-llm-rag-systems/)
    
69. Multi Needle in a Haystack - LangChain Blog, accessed May 10, 2025, [https://blog.langchain.dev/multi-needle-in-a-haystack/](https://blog.langchain.dev/multi-needle-in-a-haystack/)
    
70. Long Context vs. RAG for LLMs: An Evaluation and Revisits - arXiv, accessed May 10, 2025, [https://arxiv.org/html/2501.01880v1](https://arxiv.org/html/2501.01880v1)
    
71. KOGTI-2023/RAG-Flow: RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. - GitHub, accessed May 10, 2025, [https://github.com/KOGTI-2023/RAG-Flow](https://github.com/KOGTI-2023/RAG-Flow)
    
72. Generate grounded answers with RAG | AI Applications - Google Cloud, accessed May 10, 2025, [https://cloud.google.com/generative-ai-app-builder/docs/grounded-gen](https://cloud.google.com/generative-ai-app-builder/docs/grounded-gen)
    
73. RAG: Everything you need to know - Vectara, accessed May 10, 2025, [https://www.vectara.com/retrieval-augmented-generation-everything-you-need-to-know](https://www.vectara.com/retrieval-augmented-generation-everything-you-need-to-know)
    
74. Reducing LLM Hallucinations: A Developer's Guide - Zep, accessed May 10, 2025, [https://www.getzep.com/ai-agents/reducing-llm-hallucinations](https://www.getzep.com/ai-agents/reducing-llm-hallucinations)
    
75. Legal AI Benchmarking: Evaluating Long Context Performance for LLMs - Thomson Reuters, accessed May 10, 2025, [https://www.thomsonreuters.com/en-us/posts/innovation/legal-ai-benchmarking-evaluating-long-context-performance-for-llms/](https://www.thomsonreuters.com/en-us/posts/innovation/legal-ai-benchmarking-evaluating-long-context-performance-for-llms/)
    
76. RAG in the Era of LLMs with 10 Million Token Context Windows | F5, accessed May 10, 2025, [https://www.f5.com/company/blog/rag-in-the-era-of-llms-with-10-million-token-context-windows](https://www.f5.com/company/blog/rag-in-the-era-of-llms-with-10-million-token-context-windows)
    
77. LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs - No Silver Bullet for LC or RAG Routing - arXiv, accessed May 10, 2025, [https://arxiv.org/html/2502.09977v1](https://arxiv.org/html/2502.09977v1)
    
78. RAG vs Large Context Window LLMs: When to use which one? - The Cloud Girl, accessed May 10, 2025, [https://www.thecloudgirl.dev/blog/rag-vs-large-context-window](https://www.thecloudgirl.dev/blog/rag-vs-large-context-window)
    
79. Effective RAG evaluation: integrated metrics are all you need - Chamomile.ai, accessed May 10, 2025, [https://chamomile.ai/rag-pain-points/](https://chamomile.ai/rag-pain-points/)
    
80. Mastering RAG Evaluation: Techniques and Challenges - Tredence, accessed May 10, 2025, [https://www.tredence.com/blog/understanding-rag-systems-the-future-of-ai-interactions](https://www.tredence.com/blog/understanding-rag-systems-the-future-of-ai-interactions)
    
81. RAG Pipeline Evaluation Using DeepEval - Haystack - Deepset, accessed May 10, 2025, [https://haystack.deepset.ai/cookbook/rag_eval_deep_eval](https://haystack.deepset.ai/cookbook/rag_eval_deep_eval)
    
82. Evaluate RAG pipeline using Ragas in Python with watsonx - IBM, accessed May 10, 2025, [https://www.ibm.com/think/tutorials/ragas-rag-evaluation-python-watsonx](https://www.ibm.com/think/tutorials/ragas-rag-evaluation-python-watsonx)
    
83. Evaluation and monitoring metrics for generative AI - Azure AI Foundry | Microsoft Learn, accessed May 10, 2025, [https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-metrics-built-in](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-metrics-built-in)
    
84. RAG Triad - TruLens, accessed May 10, 2025, [https://www.trulens.org/getting_started/core_concepts/rag_triad/](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
    
85. Using the RAG Triad for RAG evaluation | DeepEval - The Open-Source LLM Evaluation Framework, accessed May 10, 2025, [https://docs.confident-ai.com/guides/guides-rag-triad](https://docs.confident-ai.com/guides/guides-rag-triad)
    
86. Mastering RAG: How To Evaluate LLMs For RAG - Galileo AI, accessed May 10, 2025, [https://www.galileo.ai/blog/how-to-evaluate-llms-for-rag](https://www.galileo.ai/blog/how-to-evaluate-llms-for-rag)
    
87. 15 Best Open-Source RAG Frameworks in 2025 - Firecrawl, accessed May 10, 2025, [https://www.firecrawl.dev/blog/best-open-source-rag-frameworks](https://www.firecrawl.dev/blog/best-open-source-rag-frameworks)
    
88. Evaluating Multi-Modal RAG - TruLens, accessed May 10, 2025, [https://www.trulens.org/cookbook/frameworks/llama_index/llama_index_multimodal/](https://www.trulens.org/cookbook/frameworks/llama_index/llama_index_multimodal/)
    
89. langchain-academy - GitHub, accessed May 10, 2025, [https://github.com/langchain-ai/langchain-academy](https://github.com/langchain-ai/langchain-academy)
    
90. Conceptual guide | 🦜️ LangChain, accessed May 10, 2025, [https://python.langchain.com/v0.2/docs/concepts/#retrieval-augmented-generation-rag](https://python.langchain.com/v0.2/docs/concepts/#retrieval-augmented-generation-rag)
    
91. Andrew-Jang/RAGHub: A community-driven collection of RAG (Retrieval-Augmented Generation) frameworks, projects, and resources. Contribute and explore the evolving RAG ecosystem. - GitHub, accessed May 10, 2025, [https://github.com/Andrew-Jang/RAGHub](https://github.com/Andrew-Jang/RAGHub)
    
92. 2024 Github Ten Best RAG Frameworks - Chief AI Sharing Circle, accessed May 10, 2025, [https://www.aisharenet.com/en/2024-github-shidaba/](https://www.aisharenet.com/en/2024-github-shidaba/)
    
93. Top 10 Open-Source RAG Frameworks you need!! - DEV Community, accessed May 10, 2025, [https://dev.to/rohan_sharma/top-10-open-source-rag-frameworks-you-need-3fhe](https://dev.to/rohan_sharma/top-10-open-source-rag-frameworks-you-need-3fhe)
    
94. langchain-ai/langchain: Build context-aware reasoning ... - GitHub, accessed May 10, 2025, [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
    
95. LlamaIndex GitHub repository insights — Restack, accessed May 10, 2025, [https://www.restack.io/docs/llamaindex-knowledge-llamaindex-github](https://www.restack.io/docs/llamaindex-knowledge-llamaindex-github)
    
96. run-llama/llama_index: LlamaIndex is the leading ... - GitHub, accessed May 10, 2025, [https://github.com/run-llama/llama_index](https://github.com/run-llama/llama_index)
    
97. deepset-ai/haystack: AI orchestration framework to build ... - GitHub, accessed May 10, 2025, [https://github.com/deepset-ai/haystack](https://github.com/deepset-ai/haystack)
    
98. GenAI: Building RAG Systems with LangChain - DEV Community, accessed May 10, 2025, [https://dev.to/ajmal_hasan/genai-building-rag-systems-with-langchain-4dbp](https://dev.to/ajmal_hasan/genai-building-rag-systems-with-langchain-4dbp)
    
99. Embedding models | 🦜️ LangChain, accessed May 10, 2025, [https://python.langchain.com/docs/integrations/text_embedding/](https://python.langchain.com/docs/integrations/text_embedding/)
    
100. LangChain - Mem0 docs, accessed May 10, 2025, [https://docs.mem0.ai/components/embedders/models/langchain](https://docs.mem0.ai/components/embedders/models/langchain)
    
101. Embedding models | 🦜️ LangChain, accessed May 10, 2025, [https://python.langchain.com/v0.2/docs/integrations/text_embedding/](https://python.langchain.com/v0.2/docs/integrations/text_embedding/)
    
102. Vector stores | 🦜️ LangChain, accessed May 10, 2025, [https://python.langchain.com/v0.2/docs/integrations/vectorstores/](https://python.langchain.com/v0.2/docs/integrations/vectorstores/)
    
103. LangChain MCP Adapters - GitHub, accessed May 10, 2025, [https://github.com/langchain-ai/langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
    
104. Comprehensive Tutorial on Building a RAG Application Using LangChain - HackerNoon, accessed May 10, 2025, [https://hackernoon.com/comprehensive-tutorial-on-building-a-rag-application-using-langchain](https://hackernoon.com/comprehensive-tutorial-on-building-a-rag-application-using-langchain)
    
105. Build a Retrieval Augmented Generation (RAG) App: Part 1 | 🦜️ LangChain, accessed May 10, 2025, [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)
    
106. What is LlamaIndex ? | IBM, accessed May 10, 2025, [https://www.ibm.com/think/topics/llamaindex](https://www.ibm.com/think/topics/llamaindex)
    
107. LlamaIndex, accessed May 10, 2025, [https://docs.llamaindex.ai/en/v0.10.33/](https://docs.llamaindex.ai/en/v0.10.33/)
    
108. Indexing & Embedding - LlamaIndex, accessed May 10, 2025, [https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing/](https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing/)
    
109. Starter Tutorial (Using OpenAI) - LlamaIndex, accessed May 10, 2025, [https://docs.llamaindex.ai/en/stable/getting_started/starter_example/](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)
    
110. Building a RAG Application Using LlamaIndex - KDnuggets, accessed May 10, 2025, [https://www.kdnuggets.com/building-a-rag-application-using-llamaindex](https://www.kdnuggets.com/building-a-rag-application-using-llamaindex)
    
111. Building RAG from Scratch (Open-source only!) - LlamaIndex, accessed May 10, 2025, [https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/)
    
112. What is the best way to scale LlamaIndex for large datasets? - Milvus, accessed May 10, 2025, [https://milvus.io/ai-quick-reference/what-is-the-best-way-to-scale-llamaindex-for-large-datasets](https://milvus.io/ai-quick-reference/what-is-the-best-way-to-scale-llamaindex-for-large-datasets)
    
113. Advanced Retrieval Strategies - LlamaIndex, accessed May 10, 2025, [https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval/](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval/)
    
114. Querying - LlamaIndex, accessed May 10, 2025, [https://docs.llamaindex.ai/en/stable/understanding/querying/querying/](https://docs.llamaindex.ai/en/stable/understanding/querying/querying/)
    
115. Building RAG from Scratch (Lower-Level) - LlamaIndex, accessed May 10, 2025, [https://docs.llamaindex.ai/en/stable/optimizing/building_rag_from_scratch/](https://docs.llamaindex.ai/en/stable/optimizing/building_rag_from_scratch/)
    
116. deepset-ai/haystack-demos - GitHub, accessed May 10, 2025, [https://github.com/deepset-ai/haystack-demos](https://github.com/deepset-ai/haystack-demos)
    
117. deepset-ai/haystack-tutorials - GitHub, accessed May 10, 2025, [https://github.com/deepset-ai/haystack-tutorials](https://github.com/deepset-ai/haystack-tutorials)
    
118. Introduction to Haystack, accessed May 10, 2025, [https://docs.haystack.deepset.ai/](https://docs.haystack.deepset.ai/)
    
119. Document Store - Haystack Documentation, accessed May 10, 2025, [https://docs.haystack.deepset.ai/docs/document-store](https://docs.haystack.deepset.ai/docs/document-store)
    
120. Haystack 2.0.0, accessed May 10, 2025, [https://haystack.deepset.ai/release-notes/2.0.0](https://haystack.deepset.ai/release-notes/2.0.0)
    
121. Integrations | Haystack - Deepset, accessed May 10, 2025, [https://haystack.deepset.ai/integrations?type=Document+Store&version=2.0](https://haystack.deepset.ai/integrations?type=Document+Store&version=2.0)
    
122. Building RAG Pipelines With Haystack and MongoDB Atlas ..., accessed May 10, 2025, [https://www.mongodb.com/developer/products/atlas/haystack-ai-mongodb-atlas-vector-demo/](https://www.mongodb.com/developer/products/atlas/haystack-ai-mongodb-atlas-vector-demo/)
    
123. Getting Started with Building RAG Systems Using Haystack - KDnuggets, accessed May 10, 2025, [https://www.kdnuggets.com/getting-started-building-rag-systems-haystack](https://www.kdnuggets.com/getting-started-building-rag-systems-haystack)
    
124. Docker - Haystack Documentation, accessed May 10, 2025, [https://docs.haystack.deepset.ai/docs/docker](https://docs.haystack.deepset.ai/docs/docker)
    
125. RAG Pipeline Evaluation Using RAGAS - Haystack - Deepset, accessed May 10, 2025, [https://haystack.deepset.ai/cookbook/rag_eval_ragas](https://haystack.deepset.ai/cookbook/rag_eval_ragas)
    
126. 7 AI Open Source Libraries To Build RAG, Agents & AI Search - DEV Community, accessed May 10, 2025, [https://dev.to/vectorpodcast/7-ai-open-source-libraries-to-build-rag-agents-ai-search-27bm?bb=190556](https://dev.to/vectorpodcast/7-ai-open-source-libraries-to-build-rag-agents-ai-search-27bm?bb=190556)
    
127. ragflow/docs/guides/dataset/run_retrieval_test.md at main - GitHub, accessed May 10, 2025, [https://github.com/infiniflow/ragflow/blob/main/docs/guides/dataset/run_retrieval_test.md](https://github.com/infiniflow/ragflow/blob/main/docs/guides/dataset/run_retrieval_test.md)
    
128. Introduction to agents | RAGFlow, accessed May 10, 2025, [https://ragflow.io/docs/dev/agent_introduction](https://ragflow.io/docs/dev/agent_introduction)
    
129. Files | RAGFlow, accessed May 10, 2025, [https://ragflow.io/docs/dev/manage_files](https://ragflow.io/docs/dev/manage_files)
    
130. A beginner's guide to building a Retrieval Augmented Generation (RAG) application from scratch, accessed May 10, 2025, [https://learnbybuilding.ai/tutorials/rag-from-scratch](https://learnbybuilding.ai/tutorials/rag-from-scratch)
    
131. Deploying RAGFlow: Best Practices & Pitfalls - Prospera Soft, accessed May 10, 2025, [https://prosperasoft.com/blog/artificial-intelligence/rag/deploying-ragflow-best-practices/](https://prosperasoft.com/blog/artificial-intelligence/rag/deploying-ragflow-best-practices/)
    
132. What are the best practices for using LlamaIndex in production? - Milvus, accessed May 10, 2025, [https://milvus.io/ai-quick-reference/what-are-the-best-practices-for-using-llamaindex-in-production](https://milvus.io/ai-quick-reference/what-are-the-best-practices-for-using-llamaindex-in-production)