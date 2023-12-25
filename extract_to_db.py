import os
import PyPDF2
import psycopg2
from dotenv import load_dotenv


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Remove NUL characters from the extracted text
                    cleaned_text = page_text.replace("\x00", "")
                    text += cleaned_text
            if not text:
                raise ValueError("No text was extracted from the PDF.")
            return text
    except Exception as e:
        raise Exception(f"An error occurred while extracting text: {str(e)}")


# Function to insert extracted text into the database
def insert_into_db(content, conn):
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO pdf_data (content) VALUES (%s)", (content,))
            conn.commit()
    except Exception as e:
        raise Exception(f"Error inserting into database: {str(e)}")


# Main function
def main():
    # Load environment variables
    load_dotenv()

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
    )

    # Get the path to the PDF file
    pdf_path = os.getenv("PDF_PATH")

    try:
        text = extract_text_from_pdf(pdf_path)
        insert_into_db(text, conn)
        print("PDF content successfully stored in the database.")
    except Exception as e:
        print(e)
    finally:
        conn.close()


# Run the main function
if __name__ == "__main__":
    main()
