
import sys
import warnings
import logging
from pathlib import Path
from typing import Annotated, List

from fastmcp import FastMCP
from mcp.server.fastmcp import Context
from langchain_openai import ChatOpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from my_llms import get_llm, get_embedding_model, SUPPORTED_LLMS, SUPPORTED_EMBEDDINGS
from ragas_singleturn import (
    score_faithfulness,
    score_answer_correctness,
    score_answer_relevance,
    score_context_precision,
    score_context_recall,
)

# Logging helpers
def log_info(ctx, msg):
    if ctx and hasattr(ctx, "log") and hasattr(ctx.log, "info"):
        ctx.log.info(msg)
    else:
        logging.info(msg)

def log_warning(ctx, msg):
    if ctx and hasattr(ctx, "log") and hasattr(ctx.log, "warning"):
        ctx.log.warning(msg)
    else:
        warnings.warn(msg)

def log_error(ctx, msg):
    if ctx and hasattr(ctx, "log") and hasattr(ctx.log, "error"):
        ctx.log.error(msg)
    else:
        logging.error(msg)

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
    retrieved_contexts: Annotated[str, "The retrieved context used to support the answer."],
    eval_framework: Annotated[str, "The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, "Identify the LLM-as-a-Judge, supported depending on LLM provider: {}".format(SUPPORTED_LLMS)],
    ctx: Context = None
) -> float:
    """
    Calculates the Ragas faithfulness score for a single response.

    Args:
        user_input: The original user question or input.
        response: The generated answer to be evaluated.
        retrieved_contexts: The retrieved context used to support the answer.
        eval_framework: The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}.
        llm: Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}.

    Returns:
        float: Faithfulness score between 0.0 (unfaithful) and 1.0 (fully faithful).

    This tool is useful for validating retrieval-augmented generation (RAG) systems or debugging hallucinations.

    Example call:
    {
        "tool": "calculate_faithfulness",
        "args": {
            "user_input": "What city is the Eiffel Tower located in?",
            "response": "The Eiffel Tower is located in Berlin.",
            "context": "The Eiffel Tower is located in Paris, France.",
            "eval_framework": "ragas",
            "llm": "gpt-4o-mini"
        }
    }
    """
    log_info(ctx, f"Starting faithfulness evaluation for user_input: {user_input}")
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        msg = (
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
        log_warning(ctx, msg)
    try:
        result = round(score_faithfulness(user_input, response, retrieved_contexts, llm=get_llm(llm)), 4)
        log_info(ctx, f"Faithfulness score computed: {result}")
        return result
    except Exception as e:
        log_error(ctx, f"Error in faithfulness evaluation: {e}")
        raise

@mcp.tool()
def calculate_answer_relevancy(
    user_input: Annotated[str, "The original user question or input."],
    response: Annotated[str, "The generated answer to be evaluated."],
    eval_framework: Annotated[str, f"The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, f"Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}"],
    embedding_model: Annotated[str, "The embedding model to use for semantic similarity, supported: {SUPPORTED_EMBEDDINGS}."],
    strictness: Annotated[int, "Number of reverse-engineered questions (default: 3)"] = 3,
    ctx: Context = None
) -> float:
    """
    Calculates the Ragas answer relevancy score for a single response.

    Args:
        user_input: The original user question or input.
        response: The generated answer to be evaluated.
        eval_framework: The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}.
        llm: Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}.
        embedding_model: The embedding model to use for semantic similarity, supported: {SUPPORTED_EMBEDDINGS}.
        strictness: Number of reverse-engineered questions (default: 3).

    Returns:
        float: Relevancy score between 0.0 and 1.0.

    This tool measures how directly the generated answer addresses the original question by
    generating reverse-engineered questions and comparing semantic similarity.

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
    log_info(ctx, f"Starting answer relevancy evaluation for user_input: {user_input}")
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        msg = (
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
        log_warning(ctx, msg)
    try:
        llm_obj = get_llm(llm)
        embedding_obj = get_embedding_model(embedding_model)
        result = round(score_answer_relevance(user_input, response, llm_obj, embedding_obj, strictness), 4)
        log_info(ctx, f"Answer relevancy score computed: {result}")
        return result
    except Exception as e:
        log_error(ctx, f"Error in answer relevancy evaluation: {e}")
        raise

@mcp.tool()
def calculate_context_precision(
    user_input: Annotated[str, "The original user question or input."],
    response: Annotated[str, "The generated answer to be evaluated."],
    retrieved_contexts: Annotated[str, "Passages retrieved for grounding."],
    eval_framework: Annotated[str, f"The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, f"Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}"],
    ctx: Context = None
) -> float:
    """
    Calculates the Ragas context precision score for retrieval evaluation.

    Args:
        user_input: The original user question or input.
        response: The generated answer to be evaluated.
        retrieved_contexts: Passages retrieved for grounding.
        eval_framework: The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}.
        llm: Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}.

    Returns:
        float: Precision score between 0.0 and 1.0.

    This tool measures the proportion of retrieved passages that are actually relevant
    by comparing against a set of reference contexts.

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
    log_info(ctx, f"Starting context precision evaluation for user_input: {user_input}")
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        msg = (
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
        log_warning(ctx, msg)
    try:
        llm_obj = get_llm(llm)
        result = round(score_context_precision(user_input, response, retrieved_contexts, llm_obj), 4)
        log_info(ctx, f"Context precision score computed: {result}")
        return result
    except Exception as e:
        log_error(ctx, f"Error in context precision evaluation: {e}")
        raise

@mcp.tool()
def calculate_context_recall(
    user_input: Annotated[str, "The original user question or input."],
    retrieved_contexts: Annotated[str, "Passages retrieved for grounding."],
    reference_answer: Annotated[str, "Ground-truth answer for recall evaluation."],
    eval_framework: Annotated[str, f"The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, f"Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}"],
    ctx: Context = None
) -> float:
    """
    Calculates the Ragas context recall score for retrieval coverage evaluation.

    Args:
        user_input: The original user question or input.
        retrieved_contexts: Passages retrieved for grounding.
        reference_answer: Ground-truth answer for recall evaluation.
        eval_framework: The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}.
        llm: Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}.

    Returns:
        float: Recall score between 0.0 and 1.0.

    This tool measures how comprehensively the retrieved passages cover the information
    contained in the ground-truth answer by verifying claim support.

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
    log_info(ctx, f"Starting context recall evaluation for user_input: {user_input}")
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        msg = (
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
        log_warning(ctx, msg)
    try:
        llm_obj = get_llm(llm)
        result = round(score_context_recall(user_input, retrieved_contexts, reference_answer, llm_obj), 4)
        log_info(ctx, f"Context recall score computed: {result}")
        return result
    except Exception as e:
        log_error(ctx, f"Error in context recall evaluation: {e}")
        raise

@mcp.tool()
def calculate_answer_correctness(
    user_input: Annotated[str, "The original user question or input."],
    response: Annotated[str, "The generated answer to be evaluated."],
    reference_answer: Annotated[str, "The reference (ground-truth) answer."],
    eval_framework: Annotated[str, f"The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}."],
    llm: Annotated[str, f"Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}"],
    embedding_model: Annotated[str, "The embedding model to use for semantic similarity, supported: {SUPPORTED_EMBEDDINGS}."],
    ctx: Context = None
) -> float:
    """
    Calculates the Ragas answer correctness score for a single QA turn.

    Args:
        user_input: The original user question or input.
        response: The generated answer to be evaluated.
        reference_answer: The reference (ground-truth) answer.
        eval_framework: The evaluation framework to use, supported: {SUPPORTED_EVAL_FRAMEWORKS}.
        llm: Identify the LLM-as-a-Judge, supported: {SUPPORTED_LLMS}.
        embedding_model: The embedding model to use for semantic similarity, supported: {SUPPORTED_EMBEDDINGS}.

    Returns:
        float: Correctness score between 0.0 and 1.0.

    This tool combines factual claim overlap and semantic similarity between the generated
    answer and the reference answer to yield a comprehensive correctness score.

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
    log_info(ctx, f"Starting answer correctness evaluation for user_input: {user_input}")
    if eval_framework not in SUPPORTED_EVAL_FRAMEWORKS:
        msg = (
            f"Unsupported evaluation framework: {eval_framework}. "
            f"Supported frameworks: {SUPPORTED_EVAL_FRAMEWORKS}. "
            f"Defaulting to {SUPPORTED_EVAL_FRAMEWORKS[0]}"
        )
        log_warning(ctx, msg)
    try:
        llm_obj = get_llm(llm)
        embedding_obj = get_embedding_model(embedding_model)
        result = round(score_answer_correctness(user_input, response, reference_answer, llm_obj, embedding_obj), 4)
        log_info(ctx, f"Answer correctness score computed: {result}")
        return result
    except Exception as e:
        log_error(ctx, f"Error in answer correctness evaluation: {e}")
        raise

if __name__ == "__main__":
    mcp.run()
    #mcp.run(
    #    transport="streamable-http",
    #    host="0.0.0.0",
    #    port=8000,
    #    path="/mcp"
    #)
