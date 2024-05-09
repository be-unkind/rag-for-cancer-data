# Retrieval Augmented Generation for data that containts information about different cancer types

## Setup

1. Install the required packages with the following line: 

    `pip install requirements.txt`

2. Update the file path if needed
3. Fill in the fields with your API keys
4. To complete data preprocessing steps, run the corresponding file with: 

    `python preprocessing.py`

5. To construct rag and get answer to your question, run the corresponding file with: 

    `python rag.py`

    You will get private/public gradio url, where you will be able to test the model using simple ui


## Structure:

### Data
- [data folder](data) - Folder with separate txt files for different cancer types
- [docs.pkl](docs.pkl) and [embeddings.pkl](embeddings.pkl) - files with saved docs and embeddings, for being able to restore the results of preprocessing step.

### Scripts
- [preprocessing.py](preprocessing.py) - Script for data preprocessing.
- [rag.py](rag.py) - Script for RAG construction.

### Notebooks
- [scrape_data.ipynb](scrape_data.ipynb) - Scripts for data scraping from the source ([link to the source](https://www.cancer.org)).
- [preprocessing.ipynb](preprocessing.ipynb) - Data preprocessing, which includes semantic chunking, text to embeddings convertion and saving to Chromadb vector database.
- [rag.ipynb](rag.ipynb) - Constructing the RAG using DSPy and connecting to language model through GROQ. Testing the model.