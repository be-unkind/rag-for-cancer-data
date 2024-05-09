import os
import chromadb

import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

import groq

import chromadb.utils.embedding_functions as embedding_functions

GROQ_API_KEY = '...'
HF_API_KEY = '...'

class GenerateAnswer(dspy.Signature):
    """Answer questions with factoid answers using relevant information from the context"""

    context = dspy.InputField(desc="May contain relevant facts.")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer based on the provided information, typically in 5-7 sentences. \
                                    Only answer if the question relates directly to the provided context \
                                    If the question does not directly match with the context, respond with 'Sorry, but I can provide you only the information about cancer.'")

class CancerRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        # self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

def construct_rag():
    llama = dspy.GROQ(model='llama3-8b-8192', api_key=GROQ_API_KEY, temperature=0, max_tokens=500)
    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=HF_API_KEY,
    model_name='BAAI/bge-small-en-v1.5'
    )

    retrieve_model = ChromadbRM(
        'cancer_data',
        os.path.join(os.getcwd(),'cancer_db'),
        embedding_function=huggingface_ef,
        k=5
    )

    dspy.settings.configure(lm=llama, rm=retrieve_model)

    uncompiled_rag = CancerRAG()
    return uncompiled_rag


if __name__ == "__main__":
    rag = construct_rag()

    test_query_1 = "What is lung carcinoid tumor?"
    response_1 = rag(test_query_1)

    print(response_1.answer)
