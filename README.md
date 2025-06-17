# RAG-Project
Experimenting with Retrieval Augmented Generation, using various Ollama chatbots

To run the code as is make sure you have the following dependencies downloaded. This will be updated the further I get.

- Ollama
- Miniconda
- ipykernel
- Python + assoc. libraries
- HTML/CSS/JS
- Some form of DB, I'm using SQLite

Currently for prototyping I will be using Llama 3.2, as it appears to be a model that runs on my Macbook Pro M3 fine, though if i get a PC set up I would like to run the Llama 4 models.

## Progress Update

I have now got a prototype model working, using the Langchain framework and tutorial to develop said prototype. This prototype contains takes input files from a supplied folder through an os.walk(), then runs through those files through an embedding program to tokenise (convert to numbers) the text in the files and simultaneously creates chunks in the documents. After this, we create a retrieval object that when fed the input chunks calculates cosine similarity scores for each of the chunks and fetches the most relevant chunks (we've set k = 3).

After successfully implementing retrieval, we combine this and selected model into the pipeline, starting up the model with a supplied template, user inputs are fed into the context retriever, which the model uses to create an natural language response. I have specified in the template to only answer questions if there is relevant context.

I currently have successfully run Llama 3.2 locally with an example output attached on the repo under the RAG_notebook.ipynb file, next step is to compile some research on RAG and build a simple pipeline. I will be uploading a folder with unit readers from different units.
