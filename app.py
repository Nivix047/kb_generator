from flask import Flask, request, jsonify
import os
import dotenv
import openai
import pinecone
import PyPDF2

app = Flask(__name__)

# Load environment variables
dotenv.load_dotenv(dotenv_path="./.env.local")

# Initialize OpenAI and Pinecone
openai.api_key = os.getenv("gpt_api_secret")
pinecone.init(api_key=os.getenv("pinecone_api_key"),
              environment="us-west1-gcp-free")

# Global variables to store data between requests
chunks = []
index = None
embed_model = "text-embedding-ada-002"

@app.route('/api/process', methods=['POST'])
def process_pdf():
    global chunks, index

    try:
        # Extract the PDF path from the request
        pdf_path = request.json.get('pdf_path')

        # Process the PDF file
        text = extract_text_from_pdf(pdf_path)

        # Break the extracted text into chunks
        chunks = break_into_chunks(text)

        # Create embeddings for the chunks and upsert to Pinecone
        index = create_and_upsert_embeddings(chunks)

        return jsonify({"message": "PDF processed and data upserted to Pinecone"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retrieve', methods=['GET'])
def retrieve_answer():
    try:
        query = request.args.get('query')
        query_with_context = generate_query_with_context(query)
        answer = complete(query_with_context)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page_obj = pdf_reader.pages[page_num]
                text += page_obj.extract_text()
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {e}")

def break_into_chunks(text, chunk_size=10000, overlap_size=5000):
    try:
        chunks = []
        for i in range(0, len(text), chunk_size - overlap_size):
            chunks.append(text[i:i + chunk_size])
        return chunks
    except Exception as e:
        raise RuntimeError(f"Error breaking text into chunks: {e}")

def create_and_upsert_embeddings(chunks):
    try:
        res = openai.Embedding.create(
            input=chunks,
            engine=embed_model
        )
        index_name = "regqa"

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension=len(res['data'][0]['embedding']),
                metric='cosine'
            )

        index = pinecone.Index(index_name=index_name)

        to_upsert = [(f"id{i}", res['data'][i]['embedding'], {"text": chunks[i]})
                     for i in range(len(res['data']))]
        index.upsert(vectors=to_upsert)

        return index
    except Exception as e:
        raise RuntimeError(f"Error creating/upserting embeddings: {e}")

def generate_query_with_context(query):
    try:
        res = openai.Embedding.create(
            input=[query],
            engine=embed_model
        )
        xq = res['data'][0]['embedding']
        res = index.query(xq, top_k=1, include_metadata=True)
        context = res['matches'][0]['metadata']['text']
        prompt = "Answer the question based on the context below.\n\ncontext:\n" + \
            context + f"\n\nQuestion: {query}\n\nAnswer:"
        return prompt
    except Exception as e:
        raise RuntimeError(f"Error generating query context: {e}")

def complete(prompt):
    try:
        res = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            temperature=0,
            max_tokens=400,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return res['choices'][0]['text'].strip()
    except Exception as e:
        raise RuntimeError(f"Error completing query: {e}")

if __name__ == '__main__':
    app.run(debug=True)
