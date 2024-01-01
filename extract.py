import os
import dotenv
import openai
import pinecone
import PyPDF2

# Load environment variables
dotenv.load_dotenv(dotenv_path=".env")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            text = ""
            # Reading each page of the PDF
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        print("Text successfully extracted from PDF.")
        return text
    except FileNotFoundError:
        print(f"Error: The file at {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None

# Function to chunk text for processing
def chunk_text(text, chunk_size=10000, overlap_size=5000):
    # Splitting text into smaller parts
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap_size)]

# Main function
def main():
    try:
        # Initialize API keys for OpenAI and Pinecone
        openai.api_key = os.getenv("OPENAI_API_SECRET")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not openai.api_key or not pinecone_api_key:
            raise ValueError("API keys not found. Please check your .env file.")

        # Extract text from the specified PDF file
        pdf_path = os.getenv("PDF_PATH")
        if not pdf_path:
            raise ValueError("PDF path not found in the environment variables.")
        text = extract_text_from_pdf(pdf_path)
        if text is None:
            raise ValueError("Failed to extract text from the PDF.")

        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")
        chunks = chunk_text(text)

        # Define embedding model and Pinecone index
        embed_model = "text-embedding-ada-002"
        index_name = "regqa"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=512, metric='cosine')
        index = pinecone.Index(index_name=index_name)

        # Process and upsert each chunk into the Pinecone index
        for i, chunk in enumerate(chunks):
            res = openai.Embedding.create(input=[chunk], engine=embed_model)
            to_upsert = [(f"id{i}", res['data'][0]['embedding'], {"text": chunk})]
            index.upsert(vectors=to_upsert)

        print("Data successfully upserted into the Pinecone index.")

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
