# Knowledge Base Generator

## Overview

This application offers a powerful solution for processing and querying information from PDF documents of any size. It utilizes Optical Character Recognition (OCR) technology to read PDF files, enabling users to extract text and ask questions related to the content of the documents.

Key features include:

1. **Embedding and Indexing:** When text is extracted from PDFs, it is first transformed into embeddings using OpenAI's powerful embedding model. This transformation converts complex text data into a vector space where semantic similarities can be efficiently computed. These embeddings are then indexed using Pinecone, a vector database specifically designed for high-speed similarity search. This indexing allows for rapid retrieval of information based on the semantic content of the query.

2. **Semantic Search:** The retrieval process leverages semantic search capabilities. When a query is submitted, it is also converted into an embedding in real-time. This query embedding is then used to search the Pinecone index for the most semantically relevant text chunks. This method ensures that the search results are contextually aligned with the query, even if the exact keywords are not present in the text.

3. **Optimizing for Contextual Relevance:** The chunking mechanism in extract.py plays a crucial role in maintaining the contextual integrity of the information. By carefully setting the chunk size and overlap size, the application ensures that each indexed segment of text is sufficiently comprehensive to provide contextually relevant answers while being concise enough for efficient processing. This balance is key to providing accurate and meaningful responses to user queries.

4. **GPT-3.5-Turbo-Instruct for Answer Generation:** Once the most relevant text chunk is retrieved, it's fed into OpenAI's GPT-3.5-Turbo-Instruct model along with the user's query. This AI model generates a response that not only answers the query but does so by considering the context provided by the retrieved text. This step ensures that the answers are not only accurate but also contextually enriched, providing a deeper understanding of the query topic.

## Installation

1. **Create and Activate a Virtual Environment (Optional, but recommended):**

- python3 -m venv venv
- source venv/bin/activate

2. **Install Dependencies:**

- pip3 install -r requirements.txt

3. **Environment Variables:**

- Create a .env file in the root directory with the following variables:
- OPENAI_API_SECRET=your_openai_api_key
- PINECONE_API_KEY=your_pinecone_api_key
- PDF_PATH=path_to_your_pdf
- PINECONE_INDEX_NAME=your_pinecone_index_name

## Usage

1. **Configure the `.env` File:**

- Before running the scripts, ensure that you have set the path to your target PDF file in your .env file. This path should be assigned to the PDF_PATH variable.

2. **Execute the Extraction Process**

- Run the `extract.py` script to start the extraction process.
- This script will read the PDF file, extract its text, and upsert the data into a Pinecone index. You will receive a success message upon successful completion of this process.

3. **Querying the Extracted Data:**

- In `retrieve.py`, locate the section where you can insert your specific query. For instance, in the provided example, the query was "What do $gt and $gte do in MongoDB?".
- After setting your query, run the `retrieve.py` script.
- This script will retrieve contexually relevant information based on your query and generate a response using GPT-3.5-Turbo-Instruct.

## Tutorial

For a comprehensive guide on how to use this application, please refer to the following tutorial video:

- [Tutorial video](https://www.youtube.com/watch?v=eOVijD9Yfpk&ab_channel=Daniel)

In this video, I demonstrate the process using a MongoDB documentation PDF from Codecademy as an example. It provides a step-by-step guide to effectively utilize the application for extracting and querying data from PDF documents.
