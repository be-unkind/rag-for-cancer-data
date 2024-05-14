import os
import chromadb

import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

import groq

import chromadb.utils.embedding_functions as embedding_functions

import gradio as gr

from configs import db_config
from configs import models_config

from dotenv import load_dotenv

load_dotenv()

class GenerateAnswerSignature(dspy.Signature):
    """
    Answer based on the provided information, typically in 5-7 sentences. 
    Only answer if the question relates directly to the topic of cancer. 
    If the question does not directly match with the context, respond with 'Sorry, but I can provide you only the information about cancer.
    """

    context = dspy.InputField(desc="May contain relevant facts.")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer based on the provided information, typically in 5-7 sentences.")

class CancerRAG(dspy.Module):
    def __init__(self, num_passages=models_config.RM_TOP_PASSAGES):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerSignature)
        # self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

def construct_rag():
    llama = dspy.GROQ(model=models_config.LM_NAME_GROQ, api_key=os.getenv('GROQ_API_KEY'), temperature=models_config.LM_TEMPERATURE, max_tokens=models_config.LM_MAX_TOKENS)
    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.getenv('HF_API_KEY'),
    model_name=models_config.EMBEDDING_MODEL_NAME
    )

    retrieve_model = ChromadbRM(
        db_config.COLLECTION_NAME,
        os.path.join(os.getcwd(), db_config.DB_NAME),
        embedding_function=huggingface_ef,
        k=models_config.RM_TOP_PASSAGES
    )

    dspy.settings.configure(lm=llama, rm=retrieve_model)

    uncompiled_rag = CancerRAG()
    return uncompiled_rag

def answer_generation(question):
    rag = construct_rag()
    answer = rag(question).answer
    return answer

def define_gradio_ui():
    iface = gr.Interface(
        fn=answer_generation,
        inputs=[gr.Textbox("What is lung carcinoid tumor?")],
        outputs="text",
        title="Cancer Question Answering",
        description="Ask a question about cancer."
        )

    iface.launch(share=True)

if __name__ == "__main__":
    define_gradio_ui()
