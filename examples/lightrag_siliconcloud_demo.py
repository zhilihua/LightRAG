import os
import asyncio
import re

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama_local import ollama_embed
from lightrag.llm.openai_local import openai_complete_if_cache
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

expr_version = 'kg_v01_lightrag'

preprocess_output_dir = os.path.join(os.path.pardir, 'outputs', 'v1_20240713')
expr_dir = os.path.join(os.path.pardir, 'experiments', expr_version)

if not os.path.exists(expr_dir):
    os.mkdir(expr_dir)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(os.path.join(os.path.pardir, 'data', '2024全球经济金融展望报告.pdf'))
documents = loader.load()

pattern = r"^全球经济金融展望报告\n中国银行研究院 \d+ 2024年"
processed_texts = '\n'.join(re.sub(pattern, '', doc.page_content) for doc in documents)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Qwen/Qwen2.5-7B-Instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("SILICONFLOW_API_KEY") or "sk-loxvhojpssdvcwkgwwdrqmvnnnpimwqfnktkrukfrilqcawp",
        base_url="https://api.siliconflow.cn/v1",
        **kwargs,
    )


# async def embedding_func(texts: list[str]) -> np.ndarray:
#     return await siliconcloud_embedding(
#         texts,
#         model="netease-youdao/bce-embedding-base_v1",
#         api_key=os.getenv("SILICONFLOW_API_KEY"),
#         max_token_size=512,
#     )


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    # result = await embedding_func(["How are you?"])
    # print("embedding_func: ", result)


asyncio.run(test_funcs())


async def initialize_rag():
    # rag = LightRAG(
    #     working_dir=expr_dir,
    #     llm_model_func=llm_model_func,
    #     embedding_func=EmbeddingFunc(
    #         embedding_dim=768, max_token_size=512, func=embedding_func
    #     ),
    # )
    rag = LightRAG(
        working_dir=expr_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text:latest", host="http://localhost:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


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


if __name__ == "__main__":
    main()