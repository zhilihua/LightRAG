import asyncio
import os
import sys
from typing import TypeVar, Union, Optional, Dict, Any
from multiprocessing.synchronize import Lock as ProcessLock
from multiprocessing import Manager

# Define a direct print function for critical logs that must be visible in all processes
# 为必须在所有进程中可见的关键日志定义直接打印功能
def direct_log(message, level="INFO", enable_output: bool = True):
    """
    Log a message directly to stderr to ensure visibility in all processes,
    including the Gunicorn master process.
    将消息直接记录到 stderr，以确保所有进程（包括 Gunicorn 主进程）的可见性。

    Args:
        message: The message to log
        level: Log level (default: "INFO")
        enable_output: Whether to actually output the log (default: True)
    """
    if enable_output:
        print(f"{level}: {message}", file=sys.stderr, flush=True)

T = TypeVar("T")
LockType = Union[ProcessLock, asyncio.Lock]

_is_multiprocess = None
_workers = None
_manager = None
_initialized = None

# shared data for storage across processes
_shared_dicts: Optional[Dict[str, Any]] = None
_init_flags: Optional[Dict[str, bool]] = None  # namespace -> initialized
_update_flags: Optional[Dict[str, bool]] = None  # namespace -> updated

# locks for mutex access
_storage_lock: Optional[LockType] = None
_internal_lock: Optional[LockType] = None
_pipeline_status_lock: Optional[LockType] = None
_graph_db_lock: Optional[LockType] = None
_data_init_lock: Optional[LockType] = None

# async locks for coroutine synchronization in multiprocess mode
_async_locks: Optional[Dict[str, asyncio.Lock]] = None

def initialize_share_data(workers: int = 1):
    """
    Initialize shared storage data for single or multi-process mode.
    为单进程或多进程模式初始化共享存储数据

    When used with Gunicorn's preload feature, this function is called once in the
    master process before forking worker processes, allowing all workers to share
    the same initialized data.
    与 Gunicorn 的预加载功能一起使用时，此函数在分叉工作进程之前在主进程中被调用一次，从而允许所有工作进程共享相同的初始化数据。

    In single-process mode, this function is called in FASTAPI lifespan function.
    在单进程模式下，该函数在FASTAPI生命周期函数中被调用。

    The function determines whether to use cross-process shared variables for data storage
    based on the number of workers. If workers=1, it uses thread locks and local dictionaries.
    If workers>1, it uses process locks and shared dictionaries managed by multiprocessing.Manager.
    该函数根据worker数量决定是否使用跨进程共享变量进行数据存储。如果workers=1，则使用线程锁和本地字典。
    如果workers>1，则使用由multiprocessing.Manager管理的进程锁和共享字典。

    Args:
        workers (int): Number of worker processes. If 1, single-process mode is used.
                      If > 1, multi-process mode with shared memory is used.
    """
    global \
        _manager, \
        _workers, \
        _is_multiprocess, \
        _storage_lock, \
        _internal_lock, \
        _pipeline_status_lock, \
        _graph_db_lock, \
        _data_init_lock, \
        _shared_dicts, \
        _init_flags, \
        _initialized, \
        _update_flags, \
        _async_locks

    # Check if already initialized
    if _initialized:
        direct_log(
            f"Process {os.getpid()} Shared-Data already initialized (multiprocess={_is_multiprocess})"
        )
        return

    _workers = workers

    if workers > 1:
        _is_multiprocess = True
        _manager = Manager()
        _internal_lock = _manager.Lock()
        _storage_lock = _manager.Lock()
        _pipeline_status_lock = _manager.Lock()
        _graph_db_lock = _manager.Lock()
        _data_init_lock = _manager.Lock()
        _shared_dicts = _manager.dict()
        _init_flags = _manager.dict()
        _update_flags = _manager.dict()

        # Initialize async locks for multiprocess mode
        _async_locks = {
            "internal_lock": asyncio.Lock(),
            "storage_lock": asyncio.Lock(),
            "pipeline_status_lock": asyncio.Lock(),
            "graph_db_lock": asyncio.Lock(),
            "data_init_lock": asyncio.Lock(),
        }

        direct_log(
            f"Process {os.getpid()} Shared-Data created for Multiple Process (workers={workers})"
        )
    else:
        _is_multiprocess = False
        _internal_lock = asyncio.Lock()
        _storage_lock = asyncio.Lock()
        _pipeline_status_lock = asyncio.Lock()
        _graph_db_lock = asyncio.Lock()
        _data_init_lock = asyncio.Lock()
        _shared_dicts = {}
        _init_flags = {}
        _update_flags = {}
        _async_locks = None  # No need for async locks in single process mode
        direct_log(f"Process {os.getpid()} Shared-Data created for Single Process")

    # Mark as initialized
    _initialized = True