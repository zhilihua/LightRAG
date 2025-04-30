import asyncio
import inspect
import logging
import os

from lightrag.base import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.lightrag import LightRAG
from lightrag.llm.ollama_local import (
    ollama_model_complete,
    ollama_embed
)
from lightrag.utils import EmbeddingFunc, setup_logger

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

setup_logger("lightrag", level="DEBUG", log_file_path="../log/lightrag.log")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="yasserrmd/Qwen2.5-7B-Instruct-1M:latest",
        llm_model_max_async=2,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=500,
            func=lambda texts: ollama_embed(
                texts, embed_model="quentinz/bge-large-zh-v1.5:latest", host="http://localhost:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)

def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Insert example text
    with open("../data/book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Test different query modes
    print("\nNaive Search:")
    print(
        rag.query("这个故事的主题是什么？", param=QueryParam(mode="naive"))
    )

    print("\nLocal Search:")
    print(
        rag.query("这个故事的主题是什么？", param=QueryParam(mode="local"))
    )

    print("\nGlobal Search:")
    print(
        rag.query("这个故事的主题是什么？", param=QueryParam(mode="global"))
    )

    print("\nHybrid Search:")
    print(
        rag.query("这个故事的主题是什么？", param=QueryParam(mode="hybrid"))
    )


if __name__ == "__main__":
    main()