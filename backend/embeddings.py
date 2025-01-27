from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    """Generate embeddings for text chunks."""
    embeddings = model.encode(chunks)
    return embeddings

def save_embeddings(embeddings, file_path):
    """Save embeddings to a file."""
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    """Load embeddings from a file."""
    return np.load(file_path)

def create_faiss_index(embeddings):
    """Create a FAISS index for embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    """Retrieve relevant chunks for a query."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks