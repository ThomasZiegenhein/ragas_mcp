from mcp_metric_server.ragas_singleturn import score_answer_correctness, score_context_recall, score_faithfulness, score_answer_relevance, score_context_precision
from mcp_metric_server.my_llms import SUPPORTED_LLMS, SUPPORTED_EMBEDDINGS, get_llm, get_embedding_model


def test_ragas_score_faithfulness():

    score = score_faithfulness(user_input = "What is color of the sky?",
                       response = "The sky is blue.", 
                       retrieved_contexts= "The sky is blue and clear.", 
                       llm=get_llm(SUPPORTED_LLMS[1]))

    assert score is not None

def test_ragas_score_answer_relevance():

    score = score_answer_relevance(user_input = "What is color of the sky?",
                       response = "The sky is blue.", 
                       llm=get_llm(SUPPORTED_LLMS[1]),
                       embedding=get_embedding_model(SUPPORTED_EMBEDDINGS[0]))

    assert score is not None

def test_ragas_score_context_precision():

    score = score_context_precision(user_input = "What is color of the sky?",
                       retrieved_contexts = "The sky is blue.", 
                       response = "The sky is blue and clear.", 
                       llm=get_llm(SUPPORTED_LLMS[1]),
                       embedding=get_embedding_model(SUPPORTED_EMBEDDINGS[0]))

    assert score is not None


def test_ragas_score_answer_correctness():

    score = score_answer_correctness(user_input = "What is color of the sky?",
                       response = "The sky is blue.", 
                       reference_answer = "The sky is blue and clear.", 
                       llm=get_llm(SUPPORTED_LLMS[1]),
                       embedding=get_embedding_model(SUPPORTED_EMBEDDINGS[0]))

    assert score is not None


