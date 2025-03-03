import os
import json
import psycopg2

# PostgreSQL connection details
DB_NAME = "pdf_data"  
DB_USER = "postgres" # I have set this to the default user
DB_PASSWORD = "123456"
DB_HOST = "localhost"
DB_PORT = "5432"

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# Ensure table `pdf_documents` exists
cur.execute("""
    CREATE TABLE IF NOT EXISTS pdf_documents (
        id SERIAL PRIMARY KEY,
        file_name TEXT UNIQUE NOT NULL,
        extracted_text TEXT NOT NULL,
        embedding VECTOR(384)  -- Ensure dimension matches your model
    );
""")
conn.commit()

# Folder where embeddings are stored
embedding_folder = "embeddings"

for json_file in os.listdir(embedding_folder):
    if json_file.endswith(".json"):
        json_path = os.path.join(embedding_folder, json_file)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filename = json_file.replace(".json", ".pdf")
        text = data["text"]
        embedding = data["embedding"]

        # Insert into database
        cur.execute("""
            INSERT INTO pdf_documents (file_name, extracted_text, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (file_name) DO UPDATE 
            SET extracted_text = EXCLUDED.extracted_text, embedding = EXCLUDED.embedding;
        """, (filename, text, embedding))

        print(f"✅ Stored embedding for {filename}")

conn.commit()
cur.close()
conn.close()
