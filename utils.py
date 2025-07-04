import logging
import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from collections import Counter


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Initialize and return the OpenAI embedding model.
    
    Returns:
        OpenAIEmbeddings: Configured embedding model instance.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=api_key
        )
        logger.info("OpenAI embeddings initialized successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
        raise


def extract_scheme_guid(sources):
    """Return the most frequent non-empty GUID from a list of documents."""
    guid_counter = Counter()
    for doc in sources:
        if doc and getattr(doc, "metadata", None):
            guid = doc.metadata.get("guid")
            if guid:
                guid_counter[str(guid).strip()] += 1
    if not guid_counter:
        return None
    return guid_counter.most_common(1)[0][0]


