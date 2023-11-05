import os

import openai
import time
from pydub import AudioSegment
from flask import Flask, request
import uuid
import xml.etree.ElementTree as ET


import json
from pydub import AudioSegment
from dotenv import load_dotenv
import requests
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

# from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
# from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from faiss_retrieve import load_vs, create_vs
from faiss_update import update_vs_new_record, delete_by_id, save_vs, update_vs_existing_record


load_dotenv()

app = Flask(__name__)



WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

TOKEN = os.environ.get('TELEGRAM_TOKEN')
API_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
BASE_URL = f"https://api.telegram.org/bot{TOKEN}/"
openai.api_key = os.environ.get('OPENAI_KEY')
API_KEY = os.environ.get('API_KEY')
SIM_THRESHOLD = 0.3

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    update = request.get_json(force=True)
    print(update)  # Print the received message to the terminal

    if 'callback_query' in update:
        handle_callback_query(update['callback_query'])
    elif 'message' in update:
        message = update['message']
        user_id = message['chat']['id']  # Extract the user's Telegram ID
        if 'text' in message:
            handle_text(message, user_id)
        elif 'voice' in message:
            handle_voice(message, user_id)

    return 'ok', 200

def handle_callback_query(callback_query):
    # Get the callback data
    callback_data = callback_query['data']
    # Send a message with the full story of the person
    chat_id = callback_query['message']['chat']['id']
        
    # Check if the callback data starts with 'full_'
    if callback_data.startswith('full_'):
        send_message(chat_id, f"😎Here comes full aggregated story of the person! Imagine how cool is that!")


def handle_text(message, user_id):
    global search_mode
    text = message['text']
    chat_id = message['chat']['id']

    if text == "/start":
        # Send the basic rules of the app
        rules = "*Welcome to DOSSIER CONNECTIONS CONCIERGE!*\n_Here how it works:_\n\n1. Send me a voice note describing a person you just met, for example: \"I've just met Sarah at hackathon, she is from Albania and works as data analyst\"\n2. To add information, simply say: \"I've met Sarah again, the one who is data analyst. She told me that she has a cute dog named \"Winnie\"\n3. To retrieve memory, describe the person you want me to give dossier on.\n\n*🦸‍♂️ Enjoy your augmented memory like super-human!*"
        send_message(chat_id, rules)
    elif text == "/search":
        search_mode = True
        send_message(chat_id, "Please send a voice message for search.")
    else:
        # Echo back the same text message
        data = {
            "chat_id": chat_id,
            "text": "Please send a voice message with a new person description, or use /search function"
        }
        response = requests.post(BASE_URL + 'sendMessage', data=data)
        print(response.status_code, response.text)

def generate_label(note_text):
    chat = ChatAnthropic(
        anthropic_api_key=API_KEY)
    template = ("You are a helpful personal assistant that helps to store informations about people I meet."
                "Analyse the following text and extract key information about the person."
                "Write the information in the structured format"
                "If some informations are missing do not write them."
                "For example:"
                "<example>"
                "Note: I met a guy called John, he lives here in London. He is a software engineer. "
                "He is 30 years old. He is married and has 2 kids. He likes to play football. He is a fan of Arsenal. "
                "Assistant:<name>John</name><age>30</age><city>London</city><job>software engineer</job><family>married, 2 kids</family><hobby>play football</hobby><interests>Arsenal</interests>"
                "</example>"
                "<example>"
                "Note: I met a woman, I dont know her name, but she is 25 years old. She is a doctor. She lives in Berkley."
                "Assistant:<age>25</age><city>Berkley</city><job>doctor</job>"
                "</example>"
                "<example>"
                "Note: I met a person called Tim in an old town of Dubrownik. He seems to be old, but I really like him. He can talk about architecture for hours. "
                "Assistant:<name>Tim</name><city>Dubrownik</city><hobby>architecture</hobby>"
                "</example>"
                "<example>"
                "Note: I met Alice in a bar. She is a student at the local University"
                "Assistant:<name>Alice</name><job>student</job><location>bar</location>"
                "")
    human_template = "Note: {text}"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    messages = chat_prompt.format_messages(text=note_text)
    model_response_label = chat(messages)
    print (model_response_label.content)

    return {"label":model_response_label.content}

def generate_story(note_text):
    chat = ChatAnthropic(
        anthropic_api_key=API_KEY)
    template = ("You are a helpful personal assistant that helps to write a very short concise dossier about person based on the provided facts and notes. Please combine them into single coherent text. Don't make up new facts")
    human_template = "Notes: {text}"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    messages = chat_prompt.format_messages(text=note_text)
    model_response_label = chat(messages)
    print (model_response_label.content)

    return (model_response_label.content)

search_mode = False

def handle_voice(message, user_id):
    global search_mode
    # Record the start time
    start_time = time.time()

    # Process voice message
    voice = message['voice']
    print("Handling voice message")

    # Download the voice message
    voice_file_id = voice['file_id']
    voice_file_path = download_file(voice_file_id, 'voice.ogg')

    # Convert the voice message to WAV format with a sample rate of 16 kHz
    audio = AudioSegment.from_ogg(voice_file_path)
    audio = audio.set_frame_rate(16000)
    audio.export('voice.wav', format='wav')
    conversion_time = time.time() - start_time

    # Transcribe the voice message with Whisper
    response = openai.Audio.transcribe(
        model="whisper-1",
        file=open('voice.wav', 'rb'),
    )

    print(response)
    response_str = json.dumps(response)
    # Parse the JSON string
    response_data = json.loads(response_str)
    # Process the transcription with GPT-3
    transcription = response_data['text']

    chat_id = message['chat']['id']
    # text = f"{response['choices'][0]['message']['content'].strip()}\n\nTotal processing time: {response_time:.2f} seconds\nFile conversion: {conversion_time:.2f} sec\nASR: {voice_time:.2f} sec\nIntent processing: {openai_time:.2f} sec"
    
    if search_mode:
        text = "Searching for: " + transcription
        send_message(chat_id, text)
        # Call your search function here
        try:
            vector_store = load_vs(user_id)
        except FileNotFoundError:
            print("Database file does not exist, creating a new one for new user")
            vector_store = create_vs(user_id, transcription)

        sim_search_results = vector_store.similarity_search_with_score(transcription, k=1)[0]
        score = sim_search_results[1]
        print(score)
        label_result = generate_label(transcription)
        print (label_result)
        json_result = json.dumps(label_result)
        if score < SIM_THRESHOLD:
            print('Found relevant person:')

            note_text = sim_search_results[0].page_content
            updated_labels = generate_label(note_text)
            # Convert the label string to an XML element
            root = ET.fromstring(f"<root>{updated_labels['label']}</root>")

            

            # Initialize an empty list to store the labels
            labels = []

            # Iterate over the child elements of the root element
            for child in root:
                # If the child element has child elements of its own (like the 'pets' element in your example),
                # iterate over its child elements
                if len(child):
                    labels.append(f"*{child.tag.capitalize()}*")  # Add the category as a headline
                    for grandchild in child:
                        # If the grandchild element has child elements of its own (like the 'pet' elements in your example),
                        # iterate over its child elements
                        if len(grandchild):
                            child_labels = []
                            for great_grandchild in grandchild:
                                # Add the great grandchild element's tag and text to the labels list
                                if great_grandchild.text is not None:  # Check if the text is not None before calling strip()
                                    child_labels.append(f"{great_grandchild.tag.capitalize()}: {great_grandchild.text.strip()}")
                            labels.append(', '.join(child_labels))  # Add the child's details as a single line
                        else:
                            # If the grandchild element doesn't have child elements, add its tag and text to the labels list
                            if grandchild.text is not None:  # Check if the text is not None before calling strip()
                                labels.append(f"{grandchild.tag.capitalize()}: {grandchild.text.strip()}")
                else:
                    # If the child element doesn't have child elements, add its tag and text to the labels list
                    if child.text is not None:  # Check if the text is not None before calling strip()
                        labels.append(f"{child.tag.capitalize()}: {child.text.strip()}")

            # Format the labels into a string
            labels_str = '\n'.join(labels)
            markdown_chars = '_'
            for char in markdown_chars:
                labels_str = labels_str.replace(char, '\\' + char)

            # # Extract the labels from the result
            # labels = label_result['label'].split('\n')
            summary = generate_story(sim_search_results[0].page_content)
            print(summary)
            # # Format the labels into a string
            # labels_str = ', '.join([f"{label.strip().split('<')[1].split('>')[0].capitalize()}: {label.strip().split('</')[0].split('>')[1]}" for label in labels if label.strip()])

            send_message(chat_id,f"Found relevant person! Please see the datapoints:\n\n{labels_str}\n\n{summary}")

        # search_documents(text, chat_id)
        print("Search activated")
        search_mode = False
        # send_message(chat_id, search_result)
    else:
        try:
            vector_store = load_vs(user_id)
        except FileNotFoundError:
            print("Database file does not exist, creating a new one for new user")
            vector_store = create_vs(user_id, transcription)

        sim_search_results = vector_store.similarity_search_with_score(transcription, k=1)[0]
        score = sim_search_results[1]
        print(score)
        print(sim_search_results[0])
        label_result = generate_label(transcription)
        print (label_result)
        json_result = json.dumps(label_result)
        if score < SIM_THRESHOLD:
            print('Found existing record:')
            # Convert the label string to an XML element
            root = ET.fromstring(f"<root>{label_result['label']}</root>")

            # Initialize an empty list to store the labels
            labels = []

            # Iterate over the child elements of the root element
            for child in root:
                # If the child element has child elements of its own (like the 'pets' element in your example),
                # iterate over its child elements
                if len(child):
                    labels.append(f"*{child.tag.capitalize()}*")  # Add the category as a headline
                    for grandchild in child:
                        # If the grandchild element has child elements of its own (like the 'pet' elements in your example),
                        # iterate over its child elements
                        if len(grandchild):
                            child_labels = []
                            for great_grandchild in grandchild:
                                # Add the great grandchild element's tag and text to the labels list
                                if great_grandchild.text is not None:  # Check if the text is not None before calling strip()
                                    child_labels.append(f"{great_grandchild.tag.capitalize()}: {great_grandchild.text.strip()}")
                            labels.append(', '.join(child_labels))  # Add the child's details as a single line
                        else:
                            # If the grandchild element doesn't have child elements, add its tag and text to the labels list
                            if grandchild.text is not None:  # Check if the text is not None before calling strip()
                                labels.append(f"{grandchild.tag.capitalize()}: {grandchild.text.strip()}")
                else:
                    # If the child element doesn't have child elements, add its tag and text to the labels list
                    if child.text is not None:  # Check if the text is not None before calling strip()
                        labels.append(f"{child.tag.capitalize()}: {child.text.strip()}")

            # Format the labels into a string
            labels_str = '\n'.join(labels)
            # # Extract the labels from the result
            # labels = label_result['label'].split('\n')
            profile_id = sim_search_results[0].metadata['id']
            # # Format the labels into a string
            # labels_str = ', '.join([f"{label.strip().split('<')[1].split('>')[0].capitalize()}: {label.strip().split('</')[0].split('>')[1]}" for label in labels if label.strip()])

            send_message(chat_id,f"Found existing record! Updating with the following data:\n\n{labels_str}",profile_id)
            vs = update_vs_existing_record(vector_store, transcription, sim_search_results[0])
            print(vs)
            print("Ahtung!")
        else:
            print('No similar records found\nUpdating database with new record')
            # Wrap the label string in a single root element and convert it to an XML element
            root = ET.fromstring(f"<root>{label_result['label']}</root>")

            # Initialize an empty list to store the labels
            labels = []

            # Iterate over the child elements of the root element
            for child in root:
                # If the child element has child elements of its own (like the 'pets' element in your example),
                # iterate over its child elements
                if len(child):
                    labels.append(f"*{child.tag.capitalize()}*")  # Add the category as a headline
                    for grandchild in child:
                        # If the grandchild element has child elements of its own (like the 'pet' elements in your example),
                        # iterate over its child elements
                        if len(grandchild):
                            child_labels = []
                            for great_grandchild in grandchild:
                                # Add the great grandchild element's tag and text to the labels list
                                child_labels.append(f"{great_grandchild.tag.capitalize()}: {great_grandchild.text.strip()}")
                            labels.append(', '.join(child_labels))  # Add the child's details as a single line
                        else:
                            # If the grandchild element doesn't have child elements, add its tag and text to the labels list
                            labels.append(f"{grandchild.tag.capitalize()}: {grandchild.text.strip()}")
                else:
                    # If the child element doesn't have child elements, add its tag and text to the labels list
                    labels.append(f"{child.tag.capitalize()}: {child.text.strip()}")

            # Format the labels into a string
            labels_str = '\n'.join(labels)
            # Send the message to the user
            send_message(chat_id, f"🙅‍♀️ No similar person found.\n\nUpdating your knowledge base with new record:\n\n{labels_str}")
            vs = update_vs_new_record(vector_store, transcription)
        save_vs(vs, user_id)
        print("FAISS updated")

    # text = transcription
    # send_message(chat_id, text)
    # if search_mode:
    #     # Call your search function here
    #     search_documents(text, chat_id)
    #     print("Search activated")
    #     search_mode = False
    #     # send_message(chat_id, search_result)
    # else:
    #     label_result = generate_label(text)
    #     print (label_result)
    #     label_text = "Great! I have saved new person in your contact book."
    #     send_message(chat_id, label_text)
    #     send_message(chat_id, label_result["label"])
    #     json_result = json.dumps(label_result)
    #     process_documents(json_result)


def download_file(file_id, filename):
    response = requests.get(BASE_URL + 'getFile', params={'file_id': file_id})
    file_info = response.json()

    if 'result' in file_info:
        file_path = file_info['result']['file_path']
        file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
        response = requests.get(file_url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        print(f"Error getting file: {file_info}")
        return None

def send_message(chat_id, text, profile_id=None):
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }

    # If a profile ID is provided, add an inline button
    if profile_id is not None:
        keyboard = [[
            {
                "text": "Show full story of this person",
                "callback_data": f"full_{profile_id}"
            }
        ]]
        data["reply_markup"] = json.dumps({"inline_keyboard": keyboard})

    response = requests.post(BASE_URL + 'sendMessage', data=data)
    print(response.status_code, response.text)

class MyDocument:
    def __init__(self, page_content, metadata={}):
        self.page_content = page_content
        self.metadata = metadata

def process_documents(documents):
    # Use the provided documents
    documents = json.loads(documents)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fs = LocalFileStore(os.path.join(base_dir, 'static', 'cache'))

    # Convert the JSON object to a list of MyDocument objects
    docs = [MyDocument(json.dumps(document), metadata={'id': str(uuid.uuid4())}) for document in documents]

    underlying_embeddings = OpenAIEmbeddings()

    embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )

   

    input_file_name_without_ext = "processed_docs"
    faiss_file_path = os.path.join(base_dir, 'static', 'faiss', f'{input_file_name_without_ext}.faiss')
    
    # Check if the FAISS index file exists
    if os.path.exists(faiss_file_path):
        # If it exists, load it
        vectorstore = FAISS.load_local(faiss_file_path, embedder)
    else:
        # If it doesn't exist, create a new FAISS index
         # Create the Chroma vector store
        vectorstore = FAISS.from_documents(documents=docs, embedding=embedder)
        list(fs.yield_keys())[:5]

    # Add new vectors to the existing index
    for doc in vectorstore.documents:
        vectorstore.add_document(doc)

    # Save the updated Faiss index
    vectorstore.save_local(faiss_file_path)
    print ("FAISS updated")

def search_documents(search, chat_id):

    # Use the provided documents
    # documents = json.loads(documents)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fs = LocalFileStore(os.path.join(base_dir, 'static', 'cache'))

    # Convert the JSON object to a list of MyDocument objects
    # docs = [MyDocument(json.dumps(documents))]

    underlying_embeddings = OpenAIEmbeddings()

    embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )

    input_file_name_without_ext = "processed_docs"
    faiss_file_path = os.path.join(base_dir, 'static', 'faiss', f'{input_file_name_without_ext}.faiss')
    
    existing_vectorstore = FAISS.load_local(faiss_file_path, embedder)
    query = search
    k = 4  # Number of documents to return
    results = existing_vectorstore.similarity_search_with_score(query, k)
    print (results)

    # Filter the results based on the score
    filtered_results = [(doc, score) for doc, score in results if score < 0.2]

    if filtered_results:
        # Format the results into a string
        results_str = "\n\n".join([f"Person {i+1}: {doc.page_content}" for i, (doc, score) in enumerate(filtered_results)])
    else:
        results_str = "No matches found. Please try again and provide a more detailed description."

    # Send the results to the user
    send_message(chat_id, results_str)

    # for doc, score in results:
    #     print(f"Document: {doc.page_content}, Score: {score}")
    # print ("FAISS searched")

    
if __name__ == '__main__':
    app.run(port=5000)
