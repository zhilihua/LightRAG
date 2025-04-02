from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, cast, final, Literal
from datetime import datetime
import os
import csv

from lightrag.base import StoragesStatus
from lightrag.operate import chunking_by_token_size
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    EmbeddingFunc,
    convert_response_to_json
)


@final
@dataclass
class LightRAG:
    """LightRAG: 简单并且快速的RAG（Retrieval-Augmented Generation）."""

    # Directory（目录）
    # ---

    working_dir: str = field(
        default=f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    """Directory where cache and temporary files are stored.（缓存和临时文件存储的目录）"""

    # Storage（存储）
    # ---

    kv_storage: str = field(default="JsonKVStorage")
    """Storage backend for key-value data.（key-value数据的存储后端）"""

    vector_storage: str = field(default="NanoVectorDBStorage")
    """Storage backend for vector embeddings.（嵌入向量的存储后端）"""

    graph_storage: str = field(default="NetworkXStorage")
    """Storage backend for knowledge graphs.（知识图的存储后端）"""

    doc_status_storage: str = field(default="JsonDocStatusStorage")
    """Storage type for tracking document processing statuses.（跟踪文档处理状态的存储类型）"""

    # Logging (Deprecated, use setup_logger in utils.py instead)
    # ---
    log_level: int | None = field(default=None)
    log_file_path: str | None = field(default=None)

    # Entity extraction（实体抽取）
    # ---

    entity_extract_max_gleaning: int = field(default=1)
    """Maximum number of entity extraction attempts for ambiguous content.（歧义内容实体提取尝试的最大次数）"""

    entity_summary_to_max_tokens: int = field(
        default=int(os.getenv("MAX_TOKEN_SUMMARY", 500))
    )

    # Text chunking（文本分块）
    # ---

    chunk_token_size: int = field(default=int(os.getenv("CHUNK_SIZE", 1200)))
    """Maximum number of tokens per text chunk when splitting documents.（切分文档时，每个文本的最大token）"""

    chunk_overlap_token_size: int = field(
        default=int(os.getenv("CHUNK_OVERLAP_SIZE", 100))
    )
    """Number of overlapping tokens between consecutive text chunks to preserve context.（相邻文档的重合token）"""

    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text.（对文本进行分块时用于tokenization的模型名称）"""

    """Maximum number of tokens used for summarizing extracted entities.（总结抽取实体时使用的最大token数）"""

    chunking_func: Callable[
        [
            str,
            str | None,
            bool,
            int,
            int,
            str,
        ],
        list[dict[str, Any]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    """
    Custom chunking function for splitting text into chunks before processing（自定义分块函数，用于在处理之前将文本拆分成块）.

    The function should take the following parameters:

        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_token_size`: The maximum number of tokens per chunk.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.
        - `tiktoken_model_name`: The name of the tiktoken model to use for tokenization.

    The function should return a list of dictionaries, where each dictionary contains the following keys:
        - `tokens`: The number of tokens in the chunk.
        - `content`: The text content of the chunk.

    Defaults to `chunking_by_token_size` if not specified.
    """

    # Node embedding（节点嵌入）
    # ---

    node_embedding_algorithm: str = field(default="node2vec")
    """Algorithm used for node embedding in knowledge graphs.（知识图谱中节点嵌入的算法）"""

    node2vec_params: dict[str, int] = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    """Configuration for the node2vec embedding algorithm:
    - dimensions: Number of dimensions for embeddings（嵌入的维数）.
    - num_walks: Number of random walks per node（每个节点的随机游动次数）.
    - walk_length: Number of steps per random walk（每次随机行走的步数）.
    - window_size: Context window size for training（训练的上下文窗口大小）.
    - iterations: Number of iterations for training（训练迭代次数）.
    - random_seed: Seed value for reproducibility（可重复性的种子值）.
    """

    # Embedding
    # ---

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use.（用于计算文本嵌入的函数。必须在使用前设置）"""

    embedding_batch_num: int = field(default=int(os.getenv("EMBEDDING_BATCH_NUM", 32)))
    """Batch size for embedding computations.（嵌入计算的批次大小）"""

    embedding_func_max_async: int = field(
        default=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", 16))
    )
    """Maximum number of concurrent embedding function calls.（最大并发嵌入函数调用数）"""

    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache（嵌入缓存的配置）.
    - enabled: If True, enables caching to avoid redundant computations.（启用缓存以避免冗余计算）
    - similarity_threshold: Minimum similarity score to use cached embeddings.（使用缓存嵌入的最小相似度得分）
    - use_llm_check: If True, validates cached embeddings using an LLM.（使用 LLM 验证缓存的嵌入）
    """

    # LLM Configuration
    # ---

    llm_model_func: Callable[..., object] | None = field(default=None)
    """Function for interacting with the large language model (LLM). Must be set before use.（与大型语言模型（LLM）交互的函数。使用前必须设置）"""

    llm_model_name: str = field(default="gpt-4o-mini")
    """Name of the LLM model used for generating responses."""

    llm_model_max_token_size: int = field(default=int(os.getenv("MAX_TOKENS", 32768)))
    """Maximum number of tokens allowed per LLM response."""

    llm_model_max_async: int = field(default=int(os.getenv("MAX_ASYNC", 4)))
    """Maximum number of concurrent LLM calls.（最大并发 LLM 调用数）"""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function.（传递给 LLM 模型函数的附加关键字参数）"""

    # Storage（存储）
    # ---

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage.（矢量数据库存储的附加参数）"""

    namespace_prefix: str = field(default="")
    """Prefix for namespacing stored data across different environments.（跨不同环境存储数据的命名空间前缀）"""

    enable_llm_cache: bool = field(default=True)
    """Enables caching for LLM responses to avoid redundant computations.（启用 LLM 响应缓存以避免冗余计算）"""

    enable_llm_cache_for_entity_extract: bool = field(default=True)
    """If True, enables caching for entity extraction steps to reduce LLM costs.（启用实体提取步骤的缓存以降低 LLM 成本）"""

    # Extensions（扩展）
    # ---

    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))
    """Maximum number of parallel insert operations.（最大并行插入操作数）"""

    addon_params: dict[str, Any] = field(
        default_factory=lambda: {
            "language": os.getenv("SUMMARY_LANGUAGE", PROMPTS["DEFAULT_LANGUAGE"])
        }
    )

    # Storages Management（存储管理）
    # ---

    auto_manage_storages_states: bool = field(default=True)
    """If True, lightrag will automatically calls initialize_storages and finalize_storages at the appropriate times.（lightrag 将在适当的时候自动调用initialize_storages 和 finalize_storages）"""

    # Storages Management（存储管理）
    # ---

    convert_response_to_json_func: Callable[[str], dict[str, Any]] = field(
        default_factory=lambda: convert_response_to_json
    )
    """
    Custom function for converting LLM responses to JSON format.（用于将 LLM 响应转换为 JSON 格式的自定义函数）

    The default function is :func:`.utils.convert_response_to_json`.（默认函数是：func:`.utils.convert_response_to_json`）
    """

    cosine_better_than_threshold: float = field(
        default=float(os.getenv("COSINE_THRESHOLD", 0.2))
    )

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)