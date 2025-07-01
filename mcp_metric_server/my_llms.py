import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv("../.env")  # Load environment variables from .env file

SUPPORTED_LLMS = ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
SUPPORTED_EMBEDDINGS = ['text-embedding-3-small']

def get_llm(llm_ident:str):
    """
    Returns the LLM instance based on the identifier.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment. Please set it in your .env file or environment variables.")

    if "gpt" in llm_ident:
        llm = ChatOpenAI(model=llm_ident, temperature=0.0, max_tokens=1024)
    else:
        raise ValueError(f"LLM identifier '{llm_ident}' is not recognized.")
    return llm

def get_embedding_model(model_name: str):
    """
    Returns the embedding model instance based on the model name.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment. Please set it in your .env file or environment variables.")

    if model_name == "text-embedding-3-small":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError(f"Embedding model '{model_name}' is not recognized.")