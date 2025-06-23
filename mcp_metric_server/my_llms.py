from langchain_openai import ChatOpenAI

SUPPORTED_LLMS = ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
SUPPORTED_EMBEDDINGS = ['text-embedding-3-small']

def get_llm(llm_ident:str):
    """
    Returns the LLM instance based on the identifier.
    """
    if "gpt" in llm_ident:
        llm = ChatOpenAI(model=llm_ident, temperature=0.0, max_tokens=1024)
    else:
        raise ValueError(f"LLM identifier '{llm_ident}' is not recognized.")
    return llm

def get_embedding_model(model_name: str):
    """
    Returns the embedding model instance based on the model name.
    """
    if model_name == "text-embedding-3-small":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError(f"Embedding model '{model_name}' is not recognized.")