import asyncio
import inspect
import os
import re

import nest_asyncio

from lightrag.base import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.lightrag import LightRAG
from lightrag.llm.ollama_local import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

nest_asyncio.apply()

expr_version = 'kg_v01_lightrag'

preprocess_output_dir = os.path.join(os.path.pardir, 'outputs', 'v1_20240713')
expr_dir = os.path.join(os.path.pardir, 'experiments', expr_version)

os.makedirs(expr_dir, exist_ok=True)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(os.path.join(os.path.pardir, 'data', '2024全球经济金融展望报告.pdf'))
documents = loader.load()

pattern = r"^全球经济金融展望报告\n中国银行研究院 \d+ 2024年"
processed_texts = '\n'.join(re.sub(pattern, '', doc.page_content) for doc in documents)

async def initialize_rag():
    rag = LightRAG(
        working_dir=expr_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name="yasserrmd/Qwen2.5-7B-Instruct-1M:latest",
        chunk_token_size=500,
        chunk_overlap_token_size=50,
        llm_model_max_async=1,
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
    rag.insert(processed_texts)

    # Test different query modes
    print("\nNaive Search:")
    print(
        rag.query(
            "报告的发布日期是什么时候？", param=QueryParam(mode="naive", top_k=2)
        )
    )

    print("\nLocal Search:")
    print(
        rag.query(
            "报告的发布日期是什么时候？", param=QueryParam(mode="local", top_k=2)
        )
    )

    print("\nGlobal Search:")
    print(
        rag.query(
            "报告的发布日期是什么时候？", param=QueryParam(mode="global", top_k=2)
        )
    )

    print("\nHybrid Search:")
    print(
        rag.query(
            "报告的发布日期是什么时候？", param=QueryParam(mode="hybrid", top_k=2)
        )
    )

    # stream response
    # resp = rag.query(
    #     "What are the top themes in this story?",
    #     param=QueryParam(mode="hybrid", stream=True),
    # )
    #
    # if inspect.isasyncgen(resp):
    #     asyncio.run(print_stream(resp))
    # else:
    #     print(resp)


if __name__ == "__main__":
    main()