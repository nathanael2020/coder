import os
import faiss
import numpy as np
from utils.config import EMBEDDING_DIM, FAISS_INDEX_BASE
from utils.config import logger
from utils.embeddings import generate_embeddings

class IndexManager:

    def __init__(self, codebase_manager):
        self.codebase_manager = codebase_manager
        self.faiss_index_file = self.codebase_manager.workspace_dir / f"{FAISS_INDEX_BASE}_{self.codebase_manager.repo_name}.bin"
        self.metadata_file = self.codebase_manager.workspace_dir / f"{FAISS_INDEX_BASE}_{self.codebase_manager.repo_name}_metadata.npy"
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.metadata = []
        
        self.full_reindex()

    def full_reindex(self):
        """Perform a full reindex of the entire codebase."""
        logger.info("Starting full reindexing of the codebase...")
        files_processed = 0
        for root, _, files in os.walk(self.codebase_manager.repo_dir):
            if self.codebase_manager.should_ignore_path(root):  # Check if the directory should be ignored
                logger.info(f"Ignoring directory: {root}")
                continue

            for file in files:
                filepath = os.path.join(root, file)
                if self.codebase_manager.should_ignore_path(filepath):  # Check if the file should be ignored
                    logger.info(f"Ignoring file: {filepath}")
                    continue

                if file.endswith(".py"):
                    logger.info(f"Processing file: {filepath}")
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            full_content = f.read()

                        embeddings = generate_embeddings(full_content)  # Generate embeddings
                        if embeddings is not None:
                            self.add_to_index(embeddings, full_content, file, filepath)
                        else:
                            logger.warning(f"Failed to generate embeddings for {filepath}")
                        files_processed += 1
                    except Exception as e:
                        logger.error(f"Error processing file {filepath}: {e}")

        self.save_index()
        logger.info(f"Full reindexing completed. {files_processed} files processed.")
        
    def clear_index(self):
        """Delete the FAISS index and metadata files if they exist, and reinitialize the index."""
        # Delete the FAISS index file
        if os.path.exists(self.faiss_index_file):
            os.remove(self.faiss_index_file)
            print(f"Deleted FAISS index file: {self.faiss_index_file}")

        # Delete the metadata file
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
            print(f"Deleted metadata file: {self.metadata_file}")

        # Reinitialize the FAISS index and metadata
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.metadata = []
        print("FAISS index and metadata cleared and reinitialized.")

    def add_to_index(self, embeddings, full_content, filename, filepath):
        if embeddings.shape[1] != self.index.d:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match FAISS index dimension {self.index.d}")

        # Convert absolute filepath to relative path
        relative_filepath = os.path.relpath(filepath, self.codebase_manager.workspace_dir)

        self.index.add(embeddings)
        self.metadata.append({
            "content": full_content,
        "filename": filename,
            "filepath": relative_filepath  # Store relative filepath
        })

    def save_index(self):
        """Save the FAISS index and metadata to disk."""
        # Convert Path to string for FAISS
        faiss_path = str(self.faiss_index_file)
        metadata_path = str(self.metadata_file)
        
        faiss.write_index(self.index, faiss_path)
        with open(metadata_path, "wb") as f:
            np.save(f, self.metadata)

    def load_index(self):
        """Load the FAISS index and metadata from disk."""
        faiss_path = str(self.faiss_index_file)
        metadata_path = str(self.metadata_file)
        
        self.index = faiss.read_index(faiss_path)
        with open(metadata_path, "rb") as f:
            self.metadata = np.load(f, allow_pickle=True).tolist()
        return self.index

    def get_metadata(self):
        return self.metadata

    def retrieve_vectors(self, n=5):
        n = min(n, self.index.ntotal)
        vectors = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        for i in range(n):
            vectors[i] = self.index.reconstruct(i)
        return vectors
    
    def is_index_loaded(self):
        logger.info(f"Checking if index is loaded: {self.index is not None}")
        return True if self.index is not None else False
    
    def inspect_metadata(self, n=5):
        metadata = self.get_metadata()
        print(f"Inspecting the first {n} metadata entries:")
        for i, data in enumerate(metadata[:n]):
            print(f"Entry {i}:")
            print(f"Filename: {data['filename']}")
            print(f"Filepath: {data['filepath']}")
            print(f"Content: {data['content'][:100]}...")  # Show the first 100 characters
            print()

_index_manager_instance = None

def get_index_manager(codebase_manager = None) -> IndexManager:
    """
    Get or create the singleton instance of IndexManager.
    Args:
        codebase_manager: The CodebaseManager instance to use for initialization
    """
    global _index_manager_instance
    
    if codebase_manager is not None:
        _index_manager_instance = IndexManager(codebase_manager)
        logger.info(f"Created new IndexManager for repo: {codebase_manager.repo_name}")
    elif _index_manager_instance is None:
        raise ValueError("codebase_manager must be provided when creating new IndexManager instance")
        
    return _index_manager_instance