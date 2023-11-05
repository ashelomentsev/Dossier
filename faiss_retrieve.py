import os

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore

OPENAI_API_KEY = os.environ.get('OPENAI_KEY')

def load_vs(telegramID):
    # load FAISS vector store
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_base = os.path.join(base_dir, 'static', 'faiss')
    index_name = f'{str(telegramID)}.faiss'  # Convert telegramID to string
    faiss_path = os.path.join(faiss_base, index_name)
    print(faiss_path)

    
    if not os.path.exists(faiss_path):
        print("No vector store found for user")
        raise FileNotFoundError(f"No vector store found for user {telegramID}")
    else:
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.load_local(faiss_base, embeddings=embedding, index_name=telegramID)

        return vector_store

def create_vs(telegramID, transcription):
    # Create a new FAISS vector store
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_base = os.path.join(base_dir, 'static', 'faiss')
    index_name = telegramID
    
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = FAISS.from_texts(transcription, embedding_function)

    # Save the vector store locally
    vector_store.save_local(faiss_base, index_name)
    print("New FAISS database created")
    print(index_name)

    return vector_store


def main():
    vector_store = load_vs()
    note = "I met this guy Alex he is German and like to party"

    result = vector_store.similarity_search(note,k=1)[0].page_content
    print(result)


if __name__ == "__main__":
    main()
