# RAG-Project
This project aims to experiment and develop AI agents and tools using Retrieval Augmented Generation (RAG), in specific a document writing tool and a basic chatbot enhanced by retrieval.

## Brief overview of RAG

Anyone who has had any level of use with LLM's like ChatGPT know that when asking a question LLM's are great at quickly coming up with an coherent and concise answer that follows the specifications of the users prompt. However, we all know that in many instances these LLM's have many instances where they tend to 'hallucinate' details in their answer, or miss some key details in the answer which makes these prompts inaccurate or unsatisfying, and when further prompted there are plenty of times where these LLM's continue to create false data to support their claims, which is dangerous if people and companies continue to increasingly depend on these AI agents.

Thus, there are many proposed ways to eliminate or reduce these hallucinations, including LLM training, dataset pruning and prompt engineering. Among the bunch is a particularly powerful tool called Retrieval Augmented Generation, which aims to fix the issue by directly supplying a dataset of knowledge to query off of instead of hiding it behind embedded layers and layers of training data.

RAG is pretty much explained by its title and can be broken down into the following sections:

- Retrieval: This is the key step. Here we supply a corpus of documents in various forms (PDF, docx, HTML, JSON) and keep them in a storage folder. The central idea is in knowing the location or source of any information provided, thus when querying an LLM, we can always request sourcing to the file, which can be compared to the source material if need be. Documents in our storage are embeeded and stored in a vector store like FAISS and then passed through to the LLM. Retrieval saves us time and resources of having to get a model trained to work on our data, we can use a prebuilt LLM and simply pass through information we want it to query on.
- Augmented Generation: We use a LLM of choice to make queries enhanced by the knowledge we've fed it. If our question is in the source we can answer it with sourcing included, otherwise we can either let the LLM answer based on prebuilt knowledge or restrict it from providing a potentially hallucinated response.


## Requirements

All requirements can be found in the YAML file, which can be used to create the conda environment needed to run the programs, we are using the Langchain framework and tutorial as my base. I am also using VSCode as my IDE of choice for running and writing these files. Feel free to add any information in the PDF folders, though currently I haven't added support for other document types, though shouldn't take too long for most (hopefully!). Also make sure Ollama is installed and that you have llama 3.2 downloaded, if you wish to have other models to use feel free to run locally but I can currently only support llama 3.2.

## Current Features

Currently we have built a history aware chatbot which has persistent storage in a session logs folder, thus can be resumed at any time. I have also made a rudimentary document writer, which can store documents as a dict and write to a word doc but loading and editing features haven't been added yet.