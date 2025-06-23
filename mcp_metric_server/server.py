import warnings
from fastmcp import FastMCP
from langchain_openai import ChatOpenAI
from typing import Annotated, List
import asyncio

from mcp_metric_server.my_llms import SUPPORTED_LLMS, get_llm, get_embedding_model
from mcp_metric_server.ragas_singleturn import score_faithfulness, score_answer_correctness, score_answer_relevance, score_context_precision, score_context_recall

# Initialize MCP server
mcp = FastMCP("LLM Metric Server")

SUPPORTED_EVAL_FRAMEWORKS = ["ragas"]

#llm = get_llm("gpt-4o-mini")  # Default LLM for evaluation
#print(llm.generate(["Say Test"]))  # Warm up the LLM to avoid cold start issues


# MCP tool wrapper
@mcp.tool()
def calculate_faithfulness(
    user_input: Annotated[str, "The original user question or input."],
    response: Annotated[str, "The generated answer to be evaluated."],
    context: Annotated[str, "The retrieved context used to support the answer."],
    eval_framework: Annotated[str, "The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, "Identify the LLM-as-a-Judge, supported: {}".format(SUPPORTED_LLMS)],
) -> float:
    """
    Calculates the Ragas faithfulness score for a single response.

    This tool evaluates whether the generated `response` is grounded in the provided `context`, based on the original `user_input`.

    Returns a float between 0.0 and 1.0:
    - `1.0` means fully faithful (grounded)
    - `0.0` means unfaithful (likely hallucinated)

    This tool is useful for validating retrieval-augmented generation (RAG) systems or debugging hallucinations.

    Example call:
    {
        "tool": "calculate_faithfulness",
        "args": {
            "user_input": "What city is the Eiffel Tower located in?",
            "response": "The Eiffel Tower is located in Berlin.",
            "context": "The Eiffel Tower is located in Paris, France."
            "eval_framework": "ragas",
            "llm": "gpt-4o-mini"
        }
    }
    """
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        warnings.warn(
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
    return round(score_faithfulness(user_input, response, context, llm=get_llm(llm)), 4)

@mcp.tool()
def calculate_answer_relevancy(
    user_input: Annotated[str, "The original user question or input."],
    response: Annotated[str, "The generated answer to be evaluated."],
    eval_framework: Annotated[str, f"The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, f"Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}"],
    embedding_model: Annotated[str, "The embedding model to use for semantic similarity, supported: {SUPPORTED_EMBEDDINGS}."],
    strictness: Annotated[int, "Number of reverse-engineered questions (default: 3)"] = 3,
) -> float:
    """
    Calculates the Ragas answer relevancy score for a single response.

    This tool measures how directly the generated answer addresses the original question by
    generating reverse-engineered questions and comparing semantic similarity.

    Returns:
        A float (often between 0.0 and 1.0) indicating alignment of answer to question.

    Example call:
    {
        "tool": "calculate_answer_relevancy",
        "args": {
            "user_input": "Why is the sky blue?",
            "response": "Because of Rayleigh scattering.",
            "eval_framework": "ragas",
            "llm": "gpt-4",
            "strictness": 3
        }
    }
    """
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        warnings.warn(
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
    llm_obj = get_llm(llm)
    embedding_obj = get_embedding_model(embedding_model)
    return round(score_answer_relevance(user_input, response, llm_obj, embedding_obj, strictness), 4)

@mcp.tool()
def calculate_context_precision(
    user_input: Annotated[str, "The original user question or input."],
    response: Annotated[str, "The generated answer to be evaluated."],
    retrieved_contexts: Annotated[str, "Passages retrieved for grounding."],
    eval_framework: Annotated[str, f"The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, f"Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}"]
) -> float:
    """
    Calculates the Ragas context precision score for retrieval evaluation.

    This tool measures the proportion of retrieved passages that are actually relevant
    by comparing against a set of reference contexts.

    Returns:
        A float between 0.0 (no retrieved passages relevant) and 1.0 (all relevant).

    Example call:
    {
        "tool": "calculate_context_precision",
        "args": {
            "user_input": "Define photosynthesis.",
            "retrieved_contexts": "...",
            "reference_contexts": "...",
            "eval_framework": "ragas",
            "llm": "gpt-4"
        }
    }
    """
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        warnings.warn(
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
    llm_obj = get_llm(llm)
    return round(score_context_precision(user_input, response, retrieved_contexts, llm_obj), 4)

@mcp.tool()
def calculate_context_recall(
    user_input: Annotated[str, "The original user question or input."],
    retrieved_contexts: Annotated[str, "Passages retrieved for grounding."],
    reference_answer: Annotated[str, "Ground-truth answer for recall evaluation."],
    eval_framework: Annotated[str, f"The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, f"Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}"],
) -> float:
    """
    Calculates the Ragas context recall score for retrieval coverage evaluation.

    This tool measures how comprehensively the retrieved passages cover the information
    contained in the ground-truth answer by verifying claim support.

    Returns:
        A float between 0.0 (no coverage) and 1.0 (full coverage).

    Example call:
    {
        "tool": "calculate_context_recall",
        "args": {
            "user_input": "Explain evolution.",
            "retrieved_contexts": "...",
            "reference_answer": "Evolution is ...",
            "eval_framework": "ragas",
            "llm": "gpt-4"
        }
    }
    """
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        warnings.warn(
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
    llm_obj = get_llm(llm)
    return round(score_context_recall(user_input, retrieved_contexts, reference_answer, llm_obj), 4)

@mcp.tool()
def calculate_answer_correctness(
    user_input: Annotated[str, "The original user question or input."],
    response: Annotated[str, "The generated answer to be evaluated."],
    reference_answer: Annotated[str, "The reference (ground-truth) answer."],
    eval_framework: Annotated[str, f"The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, f"Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}"],
    embedding_model: Annotated[str, "The embedding model to use for semantic similarity, supported: {SUPPORTED_EMBEDDINGS}."],
) -> float:
    """
    Calculates the Ragas answer correctness score for a single QA turn.

    This tool combines factual claim overlap and semantic similarity between the generated
    answer and the reference answer to yield a comprehensive correctness score.

    Returns:
        A float between 0.0 (incorrect) and 1.0 (fully correct).

    Example call:
    {
        "tool": "calculate_answer_correctness",
        "args": {
            "user_input": "What is gravity?",
            "response": "Gravity is the force that ...",
            "reference_answer": "Gravity is the attraction ...",
            "eval_framework": "ragas",
            "llm": "gpt-4",
            "weights": [0.8, 0.2]
        }
    }
    """
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        warnings.warn(
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
    llm_obj = get_llm(llm)
    embedding_obj = get_embedding_model(embedding_model)
    return round(score_answer_correctness(user_input, response, reference_answer, llm_obj, embedding_obj), 4)

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp"
    )
