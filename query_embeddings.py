import psycopg2
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load a pre-trained model and tokenizer (e.g., from Hugging Face)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to convert a query text to an embedding
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Function to search for the most similar PDFs using cosine similarity
def search_similar_pdfs(query_embedding, top_n=5):
    # Connect to your PostgreSQL database
    conn = psycopg2.connect(
        dbname="pdf_data",           # Replace with your actual database name
        user="postgres",             # Your PostgreSQL username
        password="123456",           # Your PostgreSQL password
        host="localhost",            # The host where PostgreSQL is running
        port="5432"                  # The default port for PostgreSQL
    )
    cur = conn.cursor()

    # Convert query_embedding to a vector (required by pgvector)
    query_vector = np.array(query_embedding, dtype=np.float32)

    # Execute the query to find the most similar embeddings
    cur.execute("""
        SELECT file_name, extracted_text, embedding
        FROM pdf_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_vector.tolist(), top_n))

    # Fetch results
    results = cur.fetchall()
    cur.close()
    conn.close()

    return results

if __name__ == "__main__":
    # Ask user for the query text
    print("Enter your query text:")
    query_text = input().strip()

    # Generate the embedding for the query text
    query_embedding = text_to_embedding(query_text)

    # Perform the similarity search
    similar_pdfs = search_similar_pdfs(query_embedding)
    
    if similar_pdfs:
        print("\nMost Similar PDFs:")
        for pdf in similar_pdfs:
            print(f"File: {pdf[0]}")
            print(f"Extracted Text: {pdf[1]}")
            print("-" * 40)
    else:
        print("No similar PDFs found.")
