import wikipedia as wiki
import re

def fetch_content(topic:str) -> str:
	"""
	Fetches content from wikipedia page for the given topic
	topic (str): The topic to search for on Wikipedia
	Returns: content from wikipedia page about the topic (str)
	"""
	wiki.set_lang("en")
	try:
		page= wiki.page(topic)
		return page.content

	except wiki.exceptions.DisambiguationError as e:
		print(f"[Diambiguation Error] '{topic}' is ambiguous. Options: {e.options[:3]}")
		return None
	except wiki.exceptions.PageError as e:
		print(f"[Page Error] '{topic}' does not exist.")
		return None
	except Exception as e:
		print(f"[General Error] '{topic}' encountered an error: {e}")
		return None


def clean_text(text:str) -> str:
	"""
	Cleans text of any unwanted characters or formatting
	text (str): The text to clean
	Returns: cleaned text (str)
	"""
	# section headers
	text= re.sub(r"=+", "", text) # Remove headings
	
	# references / citations
	text= re.sub(r"\[\w+\]", "", text)
	
	# remove extra whitespace
	text= re.sub(r"\s+", "", text)
	
	return text.strip() # remove leading/trailing whitespace


# Testing

# if __name__ == "__main__":
# 	topic= "Artificial Intelligence"
# 	raw= fetch_content(topic)
# 	print(raw[:1000])