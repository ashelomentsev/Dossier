import os
import faiss
import numpy as np
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import InMemoryStore, LocalFileStore, RedisStore
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.environ.get('OPENAI_KEY')

# Load the documents
base_dir = os.path.dirname(os.path.abspath(__file__))

fs = LocalFileStore(os.path.join(base_dir, 'static', 'cache'))

# Ask the user for the input file name
input_file_name = input("Please enter the input file name: ")
import_file_json = os.path.join(base_dir, 'static', 'json', input_file_name)

underlying_embeddings = OpenAIEmbeddings()
embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, fs, namespace=underlying_embeddings.model
)

# Use the custom loader with the text splitter
loader = JSONLoader(import_file_json, jq_schema='.[].note')
docs = loader.load() # List of Document objects
print(docs[0].page_content) 

# Create the Chroma vector store
vectorstore = FAISS.from_documents(documents=docs, embedding=embedder)
list(fs.yield_keys())[:5]
# Get the vectors from the Chroma vector store
# docs_vectors = [doc.vector for doc in vectorstore.documents]
# # Keep a separate list of the documents
# documents = docs

# # Convert the vectors to a 2D numpy array
# docs_vectors = np.array(docs_vectors).astype('float32')

# # Create a Faiss index and add the vectors to the index
# dimension = docs_vectors.shape[1]  # The dimension of the vectors
# index = faiss.IndexFlatL2(dimension)
# index.add(docs_vectors)
# doc_ids = [doc.unique_id for doc in vectorstore.documents]  # Assume each document has a unique id

# with open(os.path.join(base_dir, 'static', 'faiss', 'doc_ids.pkl'), 'wb') as f:
#     pickle.dump(doc_ids, f)
# Get the filename without extension
input_file_name_without_ext = os.path.splitext(input_file_name)[0]
# Save the Faiss index
vectorstore.save_local(os.path.join(base_dir, 'static', 'faiss', f'{input_file_name_without_ext}.faiss'))