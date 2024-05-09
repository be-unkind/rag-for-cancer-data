## Retrieval Augmented Generation for data that containts information about different cancer types


### Structure:
- [scrape_data.ipynb](scrape_data.ipynb) - Scripts for data scraping from the source ([link to the source](https://www.cancer.org)).
- [data folder](data) - Folder with separate txt files for different cancer types
- [preprocessing.ipynb](preprocessing.ipynb) - Data preprocessing, which includes semantic chunking, text to embeddings convertion and saving to Chromadb vector database.
- [docs.pkl](docs.pkl) and [embeddings.pkl](embeddings.pkl) - files with saved docs and embeddings, for being able to restore the results of preprocessing step.
- [rag.ipynb](rag.ipynb) - Constructing the RAG using DSPy and connecting to language model through GROQ. Testing the model.