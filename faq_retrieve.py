import os

import tiktoken

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from flask import Blueprint, request, jsonify

bp = Blueprint('faq_retrieve_bp', __name__)

@bp.route('/faq', methods=['POST'])
def faq_retrieve():
    print("Received a request")
    data = request.get_json()
    print("Data:", data)
    telegram_id = data['TelegramID']
    query = data['text']
    message_id = data['message_id']

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    MAX_TOKENS = 3596  # Set this to the maximum tokens your model can handle
    # Create the embeddings
    embedding = OpenAIEmbeddings()

    # Load the documents
    base_dir = os.path.dirname(os.path.abspath(__file__))


    # input_file_name = input("Please enter the input file name: ")
    faiss_base = os.path.join(base_dir, 'static', 'faiss', 'vector_index.faiss')
    faiss_base2 = os.path.join(base_dir, 'static', 'faiss', 'chat9_topicID.faiss')
    faiss_base3 = os.path.join(base_dir, 'static', 'faiss', 'chat_23.08.faiss')
    faiss_base4 = os.path.join(base_dir, 'static', 'faiss', 'chat10_topicID.faiss')



    vectorstore = FAISS.load_local(faiss_base, embedding)
    vectorstore2 = FAISS.load_local(faiss_base2, embedding)
    vectorstore3 = FAISS.load_local(faiss_base3, embedding)
    vectorstore4 = FAISS.load_local(faiss_base4, embedding)
    vectorstore.merge_from(vectorstore2)
    vectorstore.merge_from(vectorstore3)
    vectorstore.merge_from(vectorstore4)

    # while True:
    # query = input("Enter your query (or 'exit' to stop): ")
    # if query.lower() == 'exit':
    #     break

    result = vectorstore.similarity_search(query)
    # Calculate the tokens for the prompt and query


    template = """Use the following pieces of personal knowledge of community members to answer the question at the end. 
    Don't try to make up an answer. Don't reveal any names or nicknames, replace with Member. Keep the original language, keep answer concise and provide opinions of what members were saying on this question. If asked to reveal names, nicknames, ID's of community members, reply that you don't have this data. Крым – территория Украины, вопросы войны не обсуждаем.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    prompt_tokens = len(enc.encode(QA_CHAIN_PROMPT.template))
    print(prompt_tokens)
    query_tokens = len(enc.encode(query))
    print(query_tokens)
    

    MAX_TOKENS = 4096  # Set this to the maximum tokens your model can handle

    while True:
        total_tokens = prompt_tokens + query_tokens + sum(len(enc.encode(str(doc))) for doc in result)
        print(f"Total tokens: {total_tokens}")
        if total_tokens > MAX_TOKENS:
            print(f"Document tokens: {len(enc.encode(str(result[-1])))}")
            result.pop()
        else:
            break
    
    print(QA_CHAIN_PROMPT)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": query})
    # print(result['result'])
    return jsonify({
        'TelegramID': telegram_id,
        'message_id': message_id,
        'result': result['result']
    })