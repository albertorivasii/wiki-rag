import sqlite3
from transformers import AutoTokenizer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import nltk
from multiprocessing import Pool, cpu_count

nltk.download("punkt")

# Load tokenizer with fast backend if available
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_fast=True)

# Token count function (for parallel use)
def count_tokens(text):
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return float('inf')

# Chunking function
def split_into_chunks(text, max_tokens=512):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_len = count_tokens(sentence)
        if token_len > max_tokens:
            continue  # Skip unreasonably long sentences

        if current_length + token_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Connect to the database
db_path = "large_wikipedia.db"
con = sqlite3.connect(db_path)
cursor = con.cursor()

# Step 1: Fill token_length column in parallel
cursor.execute("SELECT id, content FROM sections")
rows = cursor.fetchall()
ids, contents = zip(*rows)

with Pool(cpu_count()) as pool:
    token_lengths = list(pool.map(count_tokens, contents))

with tqdm(total=len(ids), desc="Updating token lengths") as pbar:
    for section_id, token_len in zip(ids, token_lengths):
        cursor.execute("UPDATE sections SET token_size = ? WHERE id = ?", (token_len, section_id))
        pbar.update(1)
    con.commit()

# Step 2: Replace long rows with ≤512-token chunks
cursor.execute("SELECT COUNT(*) FROM sections WHERE token_size > 512")
long_count = cursor.fetchone()[0]

with tqdm(total=long_count, desc="Chunking long sections") as pbar:
    last_id = 0
    while True:
        cursor.execute(
            "SELECT id, topic, content FROM sections WHERE token_size > 512 AND id > ? ORDER BY id ASC LIMIT 2000",
            (last_id,)
        )
        batch = cursor.fetchall()
        if not batch:
            break

        for section_id, topic, content in batch:
            chunks = split_into_chunks(content)

            # Remove original row
            cursor.execute("DELETE FROM sections WHERE id = ?", (section_id,))

            for chunk in chunks:
                token_len = count_tokens(chunk)
                cursor.execute(
                    "INSERT INTO sections (topic, content, token_length) VALUES (?, ?, ?)",
                    (topic, chunk, token_len)
                )

            last_id = section_id
            pbar.update(1)

        con.commit()

con.close()
print("✅ Finished cleaning. All rows are now ≤ 512 tokens.")
