import psycopg2
from sentence_transformers import SentenceTransformer

# PostgreSQL connection details
DB_NAME = "pdf_data"  # Ensure this matches your database name
DB_USER = "postgres"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "5432"

# Load the Hugging Face model (same as in `generate_embeddings.py`)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate an embedding for a given query text"""
    return model.encode(text).tolist()

def search_similar_pdfs(query, top_k=5):
    """Search for the most similar PDFs in the database"""
    query_embedding = get_embedding(query)

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    # Perform similarity search using cosine distance
    cur.execute("""
        SELECT filename, text, 1 - (embedding <=> %s) AS similarity
        FROM pdf_documents
        ORDER BY similarity DESC
        LIMIT %s;
    """, (query_embedding, top_k))

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results

if __name__ == "__main__":
    query_text = input("Enter search query: ")
    results = search_similar_pdfs(query_text)

    print("\n🔍 Top Matching PDFs:")
    for filename, text, similarity in results:
        print(f"\n📄 File: {filename} (Similarity: {similarity:.4f})")
        print(f"📌 Snippet: {text[:500]}...")  # Show first 500 chars of text
