import asyncio
import configparser
import warnings
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterator, cast, final, Literal
from datetime import datetime
import os
import csv

from dotenv import load_dotenv

from lightrag.base import (
    StoragesStatus,
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
    StorageNameSpace, QueryParam
)

from lightrag.kg import verify_storage_implementation, STORAGES
from lightrag.namespace import NameSpace, make_namespace
from lightrag.operate import chunking_by_token_size, extract_entities, kg_query
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    EmbeddingFunc,
    convert_response_to_json,
    logger,
    check_storage_env_vars,
    limit_async_func_call,
    lazy_external_import,
    always_get_an_event_loop,
    compute_mdhash_id,
    clean_text,
    get_content_summary
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
# 使用当前文件夹内的 .env 允许为每个 lightrag 实例使用不同的 .env 文件，操作系统环境变量优先于 .env 文件
load_dotenv(dotenv_path=".env", override=False)

# TODO: TO REMOVE @Yannick
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

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

    # `__post_init__` 是在使用 Python 的 `dataclasses` 模块创建数据类时，可以定义的一个特殊方法。
    # 这个方法在数据类实例初始化后被调用，用于执行一些额外的初始化逻辑。
    def __post_init__(self):
        from lightrag.kg.shared_storage import (
            initialize_share_data,
        )

        # Handle deprecated parameters(处理已弃用的参数)
        if self.log_level is not None:
            warnings.warn(
                "WARNING: log_level parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )
        if self.log_file_path is not None:
            warnings.warn(
                "WARNING: log_file_path parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )

        # Remove these attributes to prevent their use
        if hasattr(self, "log_level"):
            delattr(self, "log_level")
        if hasattr(self, "log_file_path"):
            delattr(self, "log_file_path")

        initialize_share_data()

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Verify storage implementation compatibility and environment variables(验证存储实现兼容性和环境变量)
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility(验证存储实现兼容性)
            verify_storage_implementation(storage_type, storage_name)
            # Check environment variables(检查环境变量)
            check_storage_env_vars(storage_name)

        # Ensure vector_db_storage_cls_kwargs has required fields(确保 vector_db_storage_cls_kwargs 具有必填字段)
        self.vector_db_storage_cls_kwargs = {
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            **self.vector_db_storage_cls_kwargs,
        }

        # Show config
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Init LLM
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(  # type: ignore
            self.embedding_func
        )

        # Initialize all storages(初始化所有存储)
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )  # type: ignore
        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls, global_config=global_config
        )
        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls, global_config=global_config
        )

        # Initialize document status storage(初始化文档状态存储)
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            global_config=asdict(
                self
            ),  # Add global_config to ensure cache works properly
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
        )
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path"},
        )

        # Initialize document status storage(初始化文档状态存储)
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
        )

        # Directly use llm_response_cache, don't create a new object
        hashing_kv = self.llm_response_cache

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,  # type: ignore
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )

        self._storages_status = StoragesStatus.CREATED

        if self.auto_manage_storages_states:
            self._run_async_safely(self.initialize_storages, "Storage Initialization")

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    async def initialize_storages(self):
        """Asynchronously initialize the storages(异步初始化存储)"""
        if self._storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                    self.full_docs,
                    self.text_chunks,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunks_vdb,
                    self.chunk_entity_relation_graph,
                    self.llm_response_cache,
                    self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    def _run_async_safely(self, async_func, action_name=""):
        """Safely execute an async function, avoiding event loop conflicts."""
        try:
            loop = always_get_an_event_loop()
            if loop.is_running():
                task = loop.create_task(async_func())
                task.add_done_callback(
                    lambda t: logger.info(f"{action_name} completed!")
                )
            else:
                loop.run_until_complete(async_func())
        except RuntimeError:
            logger.warning(
                f"No running event loop, creating a new loop for {action_name}."
            )
            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_func())
            loop.close()

    def insert(
            self,
            input: str | list[str],
            split_by_character: str | None = None,
            split_by_character_only: bool = False,
            ids: str | list[str] | None = None,
            file_paths: str | list[str] | None = None,
    ) -> None:
        """Sync Insert documents with checkpoint support(同步插入带有检查点支持的文档)

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
        """
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(
                input, split_by_character, split_by_character_only, ids, file_paths
            )
        )

    async def ainsert(
            self,
            input: str | list[str],
            split_by_character: str | None = None,
            split_by_character_only: bool = False,
            ids: str | list[str] | None = None,
            file_paths: str | list[str] | None = None,
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
        """
        await self.apipeline_enqueue_documents(input, ids, file_paths)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

    async def apipeline_enqueue_documents(
            self,
            input: str | list[str],
            ids: list[str] | None = None,
            file_paths: str | list[str] | None = None,
    ) -> None:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs
        2. Remove duplicate contents
        3. Generate document initial status
        4. Filter out already processed documents
        5. Enqueue document in status

        Args:
            input: Single document string or list of document strings
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
        """
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # If file_paths is provided, ensure it matches the number of documents
        if file_paths is not None:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            if len(file_paths) != len(input):
                raise ValueError(
                    "Number of file paths must match the number of documents"
                )
        else:
            # If no file paths provided, use placeholder
            file_paths = ["unknown_source"] * len(input)

        # 1. Validate ids if provided or generate MD5 hash IDs
        if ids is not None:
            # Check if the number of IDs matches the number of documents
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")

            # Check if IDs are unique
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

            # Generate contents dict of IDs provided by user and documents
            contents = {
                id_: {"content": doc, "file_path": path}
                for id_, doc, path in zip(ids, input, file_paths)
            }
        else:
            # Clean input text and remove duplicates
            cleaned_input = [
                (clean_text(doc), path) for doc, path in zip(input, file_paths)
            ]
            unique_content_with_paths = {}

            # Keep track of unique content and their paths
            for content, path in cleaned_input:
                if content not in unique_content_with_paths:
                    unique_content_with_paths[content] = path

            # Generate contents dict of MD5 hash IDs and documents with paths
            contents = {
                compute_mdhash_id(content, prefix="doc-"): {
                    "content": content,
                    "file_path": path,
                }
                for content, path in unique_content_with_paths.items()
            }

        # 2. Remove duplicate contents
        unique_contents = {}
        for id_, content_data in contents.items():
            content = content_data["content"]
            file_path = content_data["file_path"]
            if content not in unique_contents:
                unique_contents[content] = (id_, file_path)

        # Reconstruct contents with unique content
        contents = {
            id_: {"content": content, "file_path": file_path}
            for content, (id_, file_path) in unique_contents.items()
        }

        # 3. Generate document initial status
        new_docs: dict[str, Any] = {
            id_: {
                "status": DocStatus.PENDING,
                "content": content_data["content"],
                "content_summary": get_content_summary(content_data["content"]),
                "content_length": len(content_data["content"]),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "file_path": content_data[
                    "file_path"
                ],  # Store file path in document status
            }
            for id_, content_data in contents.items()
        }

        # 4. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Log ignored document IDs
        ignored_ids = [
            doc_id for doc_id in unique_new_doc_ids if doc_id not in new_docs
        ]
        if ignored_ids:
            logger.warning(
                f"Ignoring {len(ignored_ids)} document IDs not found in new_docs"
            )
            for doc_id in ignored_ids:
                logger.warning(f"Ignored document ID: {doc_id}")

        # Filter new_docs to only include documents with unique IDs
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        # 5. Store status document
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")

    async def apipeline_process_enqueue_documents(
            self,
            split_by_character: str | None = None,
            split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Split document content into chunks
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )

        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Default Job",
                        "job_start": datetime.now().isoformat(),
                        "docs": 0,
                        "batchs": 0,
                        "cur_batch": 0,
                        "request_pending": False,  # Clear any previous request
                        "latest_message": "",
                    }
                )
                # Cleaning history_messages without breaking it as a shared list object
                del pipeline_status["history_messages"][:]
            else:
                # Another process is busy, just set request flag and return
                pipeline_status["request_pending"] = True
                logger.info(
                    "Another process is already processing the document queue. Request queued."
                )
                return

        try:
            # Process documents until no more documents or requests
            while True:
                if not to_process_docs:
                    log_message = "All documents have been processed or are duplicates"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                # 2. split docs into chunks, insert chunks, update doc status
                docs_batches = [
                    list(to_process_docs.items())[i: i + self.max_parallel_insert]
                    for i in range(0, len(to_process_docs), self.max_parallel_insert)
                ]

                log_message = f"Processing {len(to_process_docs)} document(s) in {len(docs_batches)} batches"
                logger.info(log_message)

                # Update pipeline status with current batch information
                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(docs_batches)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Get first document's file path and total count for job name
                first_doc_id, first_doc = next(iter(to_process_docs.items()))
                first_doc_path = first_doc.file_path
                path_prefix = first_doc_path[:20] + (
                    "..." if len(first_doc_path) > 20 else ""
                )
                total_files = len(to_process_docs)
                job_name = f"{path_prefix}[{total_files} files]"
                pipeline_status["job_name"] = job_name

                async def process_document(
                        doc_id: str,
                        status_doc: DocProcessingStatus,
                        split_by_character: str | None,
                        split_by_character_only: bool,
                        pipeline_status: dict,
                        pipeline_status_lock: asyncio.Lock,
                ) -> None:
                    """Process single document"""
                    try:
                        # Get file path from status document
                        file_path = getattr(status_doc, "file_path", "unknown_source")

                        # Generate chunks from document
                        chunks: dict[str, Any] = {
                            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                **dp,
                                "full_doc_id": doc_id,
                                "file_path": file_path,  # Add file path to each chunk
                            }
                            for dp in self.chunking_func(
                                status_doc.content,
                                split_by_character,
                                split_by_character_only,
                                self.chunk_overlap_token_size,
                                self.chunk_token_size,
                                self.tiktoken_model_name,
                            )
                        }

                        # Process document (text chunks and full docs) in parallel
                        # Create tasks with references for potential cancellation
                        doc_status_task = asyncio.create_task(
                            self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.PROCESSING,
                                        "chunks_count": len(chunks),
                                        "content": status_doc.content,
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now().isoformat(),
                                        "file_path": file_path,
                                    }
                                }
                            )
                        )
                        chunks_vdb_task = asyncio.create_task(
                            self.chunks_vdb.upsert(chunks)
                        )
                        entity_relation_task = asyncio.create_task(
                            self._process_entity_relation_graph(
                                chunks, pipeline_status, pipeline_status_lock
                            )
                        )
                        full_docs_task = asyncio.create_task(
                            self.full_docs.upsert(
                                {doc_id: {"content": status_doc.content}}
                            )
                        )
                        text_chunks_task = asyncio.create_task(
                            self.text_chunks.upsert(chunks)
                        )
                        tasks = [
                            doc_status_task,
                            chunks_vdb_task,
                            entity_relation_task,
                            full_docs_task,
                            text_chunks_task,
                        ]
                        await asyncio.gather(*tasks)
                        await self.doc_status.upsert(
                            {
                                doc_id: {
                                    "status": DocStatus.PROCESSED,
                                    "chunks_count": len(chunks),
                                    "content": status_doc.content,
                                    "content_summary": status_doc.content_summary,
                                    "content_length": status_doc.content_length,
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now().isoformat(),
                                    "file_path": file_path,
                                }
                            }
                        )
                    except Exception as e:
                        # Log error and update pipeline status
                        error_msg = f"Failed to process document {doc_id}: {str(e)}"
                        logger.error(error_msg)
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = error_msg
                            pipeline_status["history_messages"].append(error_msg)

                            # Cancel other tasks as they are no longer meaningful
                            for task in [
                                chunks_vdb_task,
                                entity_relation_task,
                                full_docs_task,
                                text_chunks_task,
                            ]:
                                if not task.done():
                                    task.cancel()
                        # Update document status to failed
                        await self.doc_status.upsert(
                            {
                                doc_id: {
                                    "status": DocStatus.FAILED,
                                    "error": str(e),
                                    "content": status_doc.content,
                                    "content_summary": status_doc.content_summary,
                                    "content_length": status_doc.content_length,
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now().isoformat(),
                                    "file_path": file_path,
                                }
                            }
                        )

                # 3. iterate over batches
                total_batches = len(docs_batches)
                for batch_idx, docs_batch in enumerate(docs_batches):
                    current_batch = batch_idx + 1
                    log_message = (
                        f"Start processing batch {current_batch} of {total_batches}."
                    )
                    logger.info(log_message)
                    pipeline_status["cur_batch"] = current_batch
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                    doc_tasks = []
                    for doc_id, status_doc in docs_batch:
                        doc_tasks.append(
                            process_document(
                                doc_id,
                                status_doc,
                                split_by_character,
                                split_by_character_only,
                                pipeline_status,
                                pipeline_status_lock,
                            )
                        )

                    # Process documents in one batch parallelly
                    await asyncio.gather(*doc_tasks)
                    await self._insert_done()

                    log_message = f"Completed batch {current_batch} of {total_batches}."
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                # Check if there's a pending request to process more documents (with lock)
                has_pending_request = False
                async with pipeline_status_lock:
                    has_pending_request = pipeline_status.get("request_pending", False)
                    if has_pending_request:
                        # Clear the request flag before checking for more documents
                        pipeline_status["request_pending"] = False

                if not has_pending_request:
                    break

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Check for pending documents again
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

        finally:
            log_message = "Document processing pipeline completed"
            logger.info(log_message)
            # Always reset busy status when done or if an exception occurs (with lock)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    async def _process_entity_relation_graph(
            self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        try:
            await extract_entities(
                chunk,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
            )
        except Exception as e:
            logger.error("Failed to extract entities and relationships")
            raise e

    async def _insert_done(
            self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.text_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    def query(
            self,
            query: str,
            param: QueryParam = QueryParam(),
            system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, system_prompt))  # type: ignore

    async def aquery(
            self,
            query: str,
            param: QueryParam = QueryParam(),
            system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        # If a custom model is provided in param, temporarily update global config
        global_config = asdict(self)

        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,  # Directly use llm_response_cache
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query.strip(),
                self.chunks_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,  # Directly use llm_response_cache
                system_prompt=system_prompt,
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,  # Directly use llm_response_cache
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response