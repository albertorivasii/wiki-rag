from transformers import AutoTokenizer
import nltk

# only need to run this once
# nltk.download('punkt')

def count_tokens(text:str, tokenizer) -> int:
	"""
	Count number of tokens in a string. Uses AutoTokenizer
	"""
	
	return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_text(text: str, tokenizer, max_tokens: int = 512) -> list:
    """
    Chunk text into smaller pieces based on max_tokens while preserving semantic meaning.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_token_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        sentence_token_count = count_tokens(sentence, tokenizer)

        if sentence_token_count > max_tokens:
            # Handle very long sentence by forcing a split at token level
            print(f"⚠️ Long sentence ({sentence_token_count} tokens). Forcibly splitting.")
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            for i in range(0, len(token_ids), max_tokens):
                chunk = tokenizer.decode(token_ids[i:i + max_tokens], skip_special_tokens=True)
                chunks.append(chunk.strip())
            continue  # Skip adding this to current_chunk

        if current_token_count + sentence_token_count <= max_tokens:
            current_chunk += " " + sentence
            current_token_count += sentence_token_count
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_token_count = sentence_token_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# testing
if __name__ == "__main__":
	from wikipedia_api import *
	tokenizer= AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
	raw = fetch_content("Artificial Intelligence")
	test_sent= "This is a sentence. It should tokenize well."
	print(f"Number of tokens: {count_tokens(test_sent, tokenizer)}")
	if raw:
		clean= clean_text(raw)
		chunks= chunk_text(clean, tokenizer, max_tokens=512)
		print(f"Number of chunks: {len(chunks)}")
		for i, chunk in enumerate(chunks):
			print(f"Chunk {i+1}: {count_tokens(chunk, tokenizer)} tokens")
			print(f"Chunk {i+1}: {chunk[0:300]}")