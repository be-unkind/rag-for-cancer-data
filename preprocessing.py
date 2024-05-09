from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
import torch
import pickle

import chromadb
import uuid

DATA_DIR = os.path.join(os.getcwd(), 'data')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def check_file_in_directory(directory, filename):
    filepath = os.path.join(directory, filename)
    return os.path.exists(filepath)

def get_raw_data():
    texts = {}

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)

        with open(filepath) as file:
            text = file.read()
            texts[f'{filename.split(".")[0]}'] = text

    keys_to_delete = []
    for key, value in texts.items():
        if value == '':
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del texts[key]

    texts_lst = list(texts.values())
    return texts_lst

def split_semantic_chunking(texts_lst, splitter_model):
    docs = splitter_model.create_documents(texts_lst)
    
    return docs

def text_to_embeddings(docs, model_embedding):
    embeddings = model_embedding.embed_documents([x.page_content for x in docs])

    return embeddings

def insert_record(collection, chunk, embedding):
    collection.add(
        embeddings=[embedding],
        documents=[chunk],
        ids=[str(uuid.uuid4())],
    )

def save_to_vector_db(docs, embeddings, db_name='cancer_db', collection_name='cancer_data'):
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), db_name))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for chunk, embedding in zip(docs, embeddings):
        insert_record(collection, chunk.page_content, embedding)

    return None

def find_embedding(model, query: str, n_results=3, db_name='cancer_db', collection_name='cancer_data'):
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), db_name))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    query_vector = model.embed_query(query)
    query_result = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
    )

    examples = ""

    for res in query_result['documents'][0]:
        try:
            examples += f"Result:\n{res}\n"
            examples += "-"*20 + '\n'
        except:
            continue

    return examples

def main(splitter_model, model_embedding):
    if check_file_in_directory(os.getcwd(), 'docs.pkl'):
        with open('/Users/nastya/Documents/litslink/cancer/docs.pkl', 'rb') as f:
            docs = pickle.load(f)
    else:
        docs = split_semantic_chunking(texts_lst, splitter_model)

    if check_file_in_directory(os.getcwd(), 'embeddings.pkl'):
        with open('/Users/nastya/Documents/litslink/cancer/embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = text_to_embeddings(docs)

    save_to_vector_db(docs, embeddings)

    return None

if __name__ == "__main__":
    texts_lst = get_raw_data()

    splitter_model = SemanticChunker(HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': DEVICE}), breakpoint_threshold_type="interquartile")
    model_embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': DEVICE})

    main(splitter_model, model_embedding)

    # Test query from db
    examples = find_embedding(model_embedding, 'What is bone cancer?', n_results=1)
    print(examples)