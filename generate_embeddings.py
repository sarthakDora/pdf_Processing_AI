import os
import json
from pdf_processing import process_pdf  # my function to extract text from PDFs
from sentence_transformers import SentenceTransformer

# Load the Hugging Face model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate an embedding for a given text"""
    return model.encode(text).tolist()

if __name__ == "__main__":
    pdf_folder = "pdfs"
    embedding_folder = "embeddings"

    os.makedirs(embedding_folder, exist_ok=True)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)

            # here I have used the process_pdf function from pdf_processing.py
            text = process_pdf(pdf_path)

            embedding = get_embedding(text)

            embedding_filename = os.path.join(embedding_folder, pdf_file.replace(".pdf", ".json"))
            with open(embedding_filename, "w", encoding="utf-8") as f:
                json.dump({"text": text, "embedding": embedding}, f)

            print(f" Extracted text & generated embedding for {pdf_file}")
