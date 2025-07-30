
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))
print(sys.path)
from ragas_singleturn import score_answer_correctness, score_context_recall, score_faithfulness, score_answer_relevance, score_context_precision
from my_llms import SUPPORTED_LLMS, SUPPORTED_EMBEDDINGS, get_llm, get_embedding_model


def test_ragas_score_faithfulness():

    score = score_faithfulness(user_input = "What is color of the sky?",
                       response = "The sky is blue.", 
                       retrieved_contexts= "The sky is blue and clear.", 
                       llm=get_llm())

    assert score is not None

def test_ragas_score_answer_relevance():

    score = score_answer_relevance(user_input = "What is color of the sky?",
                       response = "The sky is blue.", 
                       llm=get_llm(),
                       embedding=get_embedding_model())

    assert score is not None

def test_ragas_score_context_precision():

    score = score_context_precision(user_input = "What is color of the sky?",
                       retrieved_contexts = "The sky is blue.", 
                       response = "The sky is blue and clear.", 
                       llm=get_llm())

    assert score is not None


def test_ragas_score_answer_correctness():

    score = score_answer_correctness(user_input = "What is color of the sky?",
                       response = "The sky is blue.", 
                       reference_answer = "The sky is blue and clear.", 
                       llm=get_llm(),
                       embedding=get_embedding_model())

    assert score is not None


