# Retrieval Augmented Generation for data that containts information about different cancer types

## Setup

1. Python

    Make sure Python is installed on your system

2. Setting up environment

    Execute the following lines in the terminal to switch to your project folder (change path to the path to your local project directory) and create virtual environment

    `cd path/to/project/folder`

    `python -m venv rag_for_cancer_data_venv`

    To deactivate the virtual environment, simply run the following line in the terminal:

    `deactivate`

3. Install the required packages with the following line: 

    `pip install -r requirements.txt`

4. Update the file paths if needed
5. Setting up Environment Variables

    - In the root directory of your project, create a file named `.env`.
    - Add your credentials

        Open the `.env` file in a text editor and add your API keys:
        ```
        GROQ_API_KEY=your_groq_api_key_here
        HF_API_KEY=your_hf_api_key_here
        ```

6. To complete data preprocessing steps, run the corresponding file with: 

    `python preprocessing.py`

7. To construct rag and get answer to your question, run the corresponding file with: 

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