import numpy as np
from core.codebase_manager import get_codebase_manager
from utils.embeddings import generate_embeddings
from utils.config import logger
import os

def search_code(query, k=5):
    """Search the FAISS index using a text query."""

    codebase_manager = get_codebase_manager()
    index_manager = codebase_manager.get_index_manager()
    
    # Ensure index is properly loaded
    if not index_manager.is_index_loaded():
        logger.warning("Index not properly loaded")
        return [{
            "filename": "NO_INDEX",
            "filepath": "",
            "content": "The index is not properly loaded. Please reindex the codebase.",
            "distance": 0.0
        }]

    if not index_manager.get_metadata() or len(index_manager.get_metadata()) == 0:
        logger.warning("Index is empty")
        return [{
            "filename": "EMPTY_INDEX",
            "filepath": "",
            "content": "The index is empty. Please index some files first.",
            "distance": 0.0
        }]

    # Generate query embedding
    query_embedding = generate_embeddings(query)
    if query_embedding is None:
        raise ValueError("Failed to generate query embedding")

    # Search the index
    distances, indices = index_manager.load_index().search(
        query_embedding, 
        min(k, index_manager.index.ntotal)
    )

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(index_manager.get_metadata()):
            metadata = index_manager.get_metadata()[idx]
            results.append({
                "filename": metadata["filename"],
                "filepath": metadata["filepath"],
                "content": metadata["content"],
                "distance": float(distances[0][i])
            })

    return results

    # except Exception as e:
    #     logger.error(f"Error during code search: {str(e)}")
    #     return [{
    #         "filename": "ERROR",
    #         "filepath": "",
    #         "content": f"An error occurred during search: {str(e)}",
    #         "distance": 0.0
    #     }]
