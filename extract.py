import os
import dotenv
import openai
import pinecone
import PyPDF2

# Load environment variables from .env file
dotenv.load_dotenv(dotenv_path=".env")

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        print("Text sucessfully extracted from PDF.")
        return text
    except FileNotFoundError:
        print(f"Error: The file at {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None

def main():
    try:
        # Initialize OpenAI with the API key
        openai.api_key = os.getenv("OPENAI_API_SECRET")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please check your .env file.")

        # Extract text from the PDF file
        pdf_path = os.getenv("PDF_PATH")  # Get the path from the environment variable
        if not pdf_path:
            raise ValueError("PDF path not found in the environment variables.")
        text = extract_text_from_pdf(pdf_path)
        if text is None:
            raise ValueError("Failed to extract text from the PDF.")

        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key not found. Please check your .env file.")
        pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")

        # Create embeddings for the text
        embed_model = "text-embedding-ada-002"
        res = openai.Embedding.create(input=[text], engine=embed_model)

        # Check if Pinecone index exists, create if not
        index_name = "regqa"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=len(res['data'][0]['embedding']), metric='cosine')

        # Create Pinecone index
        index = pinecone.Index(index_name=index_name)

        # Upsert text into the index
        to_upsert = [("id", res['data'][0]['embedding'], {"text": text})]
        index.upsert(vectors=to_upsert)

        # Save index name to environment for retrieval
        os.environ["PINECONE_INDEX_NAME"] = index_name

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
