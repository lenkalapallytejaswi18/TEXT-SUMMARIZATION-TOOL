import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Download required NLTK data safely
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def summarize_text(text, num_sentences=3):
    # Use Sumy's built-in Tokenizer, not NLTK's directly
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

if __name__ == "__main__":
    text_input = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence
    concerned with the interactions between computers and human language. In particular, it focuses on how to
    program computers to process and analyze large amounts of natural language data. The result is a computer capable
    of understanding the contents of documents, including the contextual nuances of the language within them.
    """
    num_sentences = 2
    summary = summarize_text(text_input, num_sentences)
    print("\nSummary:\n", summary)
