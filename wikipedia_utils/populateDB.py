import sqlite3 as lite
from datasets import load_dataset
import pandas as pd

def connect_db(db_name):
    """Connect to SQLite database."""
    conn = lite.connect(db_name)
    cursor = conn.cursor()
    return conn, cursor

def create_tables(cursor):
    """Create the database tables."""
    create_sections_table = """
    CREATE TABLE IF NOT EXISTS sections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        content TEXT,
        sentiment TEXT
    )
    """
    create_url_table = """
    CREATE TABLE IF NOT EXISTS url_table (
        topic TEXT,
        url TEXT
    )
    """
    cursor.execute(create_sections_table)
    cursor.execute(create_url_table)

def process_and_upload(cursor, chunk):
    """Process a chunk of data and upload it to the database."""
    for _, row in chunk.iterrows():
        topic = row["title"]
        text = row["text"]
        url = row["url"]

        # Insert into url_table
        cursor.execute("INSERT INTO url_table (topic, url) VALUES (?, ?)", (topic, url))

        # Split text into sections and insert into sections table
        sections = split_into_sections(text)
        for section in sections:
            cursor.execute(
                "INSERT INTO sections (topic, content, sentiment) VALUES (?, ?, ?)",
                (topic, section, "neutral"),
            )

def split_into_sections(text):
    """Split text into sections using \n\n as the delimiter."""
    return text.split("\n\n")

if __name__ == "__main__":
    db_name = "large_wikipedia.db"

    # Connect to the database
    conn, cursor = connect_db(db_name)

    # Create tables
    create_tables(cursor)
    conn.commit()

    # Load the dataset in chunks
    dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    chunk_size = 1_000_000  # Process 1000 rows at a time

    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i + chunk_size]
        chunk_df= pd.DataFrame(chunk)
        process_and_upload(cursor, chunk_df)
        conn.commit()  # Commit after processing each chunk
        print(f"Processed rows {i} to {i + chunk_size - 1}")

    # Close the connection
    conn.close()
