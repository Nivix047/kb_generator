import os
import dotenv
import openai
import pinecone
import PyPDF2

# Load environment variables
dotenv.load_dotenv(dotenv_path="./.env")

# Initialize OpenAI and Pinecone with API keys
openai.api_key = os.getenv("OPENAI_API_SECRET")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp-free")

# Display Pinecone user details
print(pinecone.whoami())

# Define paths and queries
PDF_PATH = "/Users/nivix047/Desktop/mongoCRUD.pdf"
QUERY = "What do $gt and $gte do in MongoDB?"

# Function to generate completions using OpenAI
def generate_completion(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response['choices'][0]['text'].strip()

# Function to retrieve text based on a query
def retrieve_text(query, embed_model, index):
    response = openai.Embedding.create(input=[query], engine=embed_model)
    query_embedding = response['data'][0]['embedding']
    search_result = index.query(query_embedding, top_k=1, include_metadata=True)
    context = search_result['matches'][0]['metadata']['text']
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return prompt

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Extract text from PDF
extracted_text = extract_text_from_pdf(PDF_PATH)

# Break extracted text into chunks for embedding
CHUNK_SIZE = 10000
OVERLAP_SIZE = 5000
chunks = [extracted_text[i:i + CHUNK_SIZE] for i in range(0, len(extracted_text), CHUNK_SIZE - OVERLAP_SIZE)]

# Create embeddings for the chunks
EMBED_MODEL = "text-embedding-ada-002"
embedding_response = openai.Embedding.create(input=chunks, engine=EMBED_MODEL)

# Define Pinecone index details and create index
INDEX_NAME = "regqa"
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=len(embedding_response['data'][0]['embedding']), metric='cosine')

index = pinecone.Index(index_name=INDEX_NAME)

# Display Pinecone index statistics
print(index.describe_index_stats())

# Upsert vectors into the index
to_upsert = [(f"id{i}", embedding_response['data'][i]['embedding'], {"text": chunks[i]}) for i in range(len(embedding_response['data']))]
index.upsert(vectors=to_upsert)

# Perform query retrieval and generate completion
query_context = retrieve_text(QUERY, EMBED_MODEL, index)
completion = generate_completion(query_context)
print(completion)

# Delete Pinecone index after use
pinecone.delete_index(INDEX_NAME)
