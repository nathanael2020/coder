import os
import numpy as np
from openai import OpenAI
from utils.config import logger
import traceback

# Load configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default_api_key")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE_URL)

def generate_embeddings(text):
    """Generate embeddings using the updated OpenAI API."""

    logger.info(f"Generating embeddings for text: {text}")
    try:
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=[text]  # Input should be a list of strings
        )
        # Extract the embedding from the response
        embeddings = response.data[0].embedding
        return np.array(embeddings).astype('float32').reshape(1, -1)
    except Exception as e:
        logger.error(f"Error generating embeddings with OpenAI: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
