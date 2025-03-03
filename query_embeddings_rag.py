import psycopg2
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import re

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

# Function to extract the best snippet matching the query
def extract_best_snippet(text, query_embedding):
    # Break the text into sentences or paragraphs
    sentences = re.split(r'\n|\. ', text)  # Split by newline or period
    best_score = -1
    best_snippet = ""
    
    for sentence in sentences:
        # Convert each sentence to an embedding
        sentence_embedding = text_to_embedding(sentence)
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(query_embedding, sentence_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(sentence_embedding))
        
        # Track the sentence with the highest similarity score
        if cosine_similarity > best_score:
            best_score = cosine_similarity
            best_snippet = sentence
    
    return best_snippet

if __name__ == "__main__":
    # Ask user for the query text
    print("Enter your query text:")
    query_text = input().strip()

    # Generate the embedding for the query text
    query_embedding = text_to_embedding(query_text)

    # Perform the similarity search
    similar_pdfs = search_similar_pdfs(query_embedding)
    
    if similar_pdfs:
        print("\nAnswering your query based on the most relevant document:")
        for pdf in similar_pdfs:
            print(f"File: {pdf[0]}")
            print(f"Extracted Text: {extract_best_snippet(pdf[1], query_embedding)}")
            print("-" * 40)
    else:
        print("No similar PDFs found.")
