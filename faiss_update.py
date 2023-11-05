from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os


def save_vs(vs: FAISS, user_id):
    # Create a new FAISS vector store
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_base = os.path.join(base_dir, 'static', 'faiss')
    index_name = user_id
    vs.save_local(faiss_base, index_name)
    print("FAISS database saved with updated record")


def delete_by_id(vs: FAISS, id):
    vs.delete([id])

    return vs


def update_vs_new_record(vector_store: FAISS, transcription):
    # search_result_id = sim_search_results.metadata['id']
    vector_store.add_texts([transcription], metadatas=[{"id": len(vector_store.docstore._dict) + 1}])
    print("vectors added")

    return vector_store


def update_vs_existing_record(vector_store: FAISS, transcription, sim_search_results):
    # update existing record
    # search_result_id = sim_search_results.metadata["id"]
    # Convert the vector store to a retriever
    # update existing record
    search_result_id = sim_search_results.metadata['id']
    print(sim_search_results)
    retriever = vector_store.as_retriever(search_kwargs={'filter': {'id': search_result_id}})
    # Read the existing record into memory
    existing_record = sim_search_results.page_content
    print("Follows existing record:")
    print(existing_record)

    # Append new data to the existing record
    updated_record = existing_record + "\n\n" + transcription
    print(updated_record)

    try:
    # Delete the old record
        vector_store.delete([search_result_id])
    except ValueError:
        # Ignore the error if the ID does not exist
        pass

    # Add the updated record as a new record
    vector_store.add_texts([updated_record], metadatas=[{"id": len(vector_store.docstore._dict) + 1}])


    # Convert the vector store to a retriever
    
    # vector_store.add_texts([transcription], ids={'id': search_result_id})
    # print(retriever)
    print("Retriever found the record and returning it")
    # Perform a search to retrieve the updated record
    # updated_record = retriever.search(new_record)



    # if vector_store.delete([id for id, doc in vector_store.docstore._dict.items() if doc.metadata["id"] == search_result_id]):
    #     new_record = sim_search_results.page_content + "\n\n" + transcription
    #     vector_store.add_texts([new_record],metadatas=[{"id": len(vector_store.docstore._dict) + 1}])

    return vector_store
