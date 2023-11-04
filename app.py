import os
from telegram import Update, Voice, Document
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import openai
import time
import psycopg2
import base64
from pydub import AudioSegment
from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import urlparse, urlunparse
from datetime import datetime
import json
import logging
from pydub import AudioSegment
import math
import requests
# from google.cloud import speech
# from google.cloud.speech import types
import io

app = Flask(__name__)
# socketio = SocketIO(app, async_mode='eventlet')

# Heroku provides the DATABASE_URL environment variable for connecting to your database.
DATABASE_URL = os.environ['DATABASE_URL']
conn = psycopg2.connect(DATABASE_URL, sslmode='require')

# When using SQLAlchemy with postgresql driver and psycopg2,
# it requires the URL to start with "postgresql://" instead of "postgres://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String(50))
    language = db.Column(db.String(20))

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_message = db.Column(db.Text)
    bot_response = db.Column(db.Text)

@app.route('/')
def index():
    return render_template('index.html')

# @socketio.on('audio')
# def transcribe(audio_data):
#     client = speech.SpeechClient()
#     audio = types.RecognitionAudio(content=audio_data)
#     config = types.RecognitionConfig(
#         encoding=types.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=16000,
#         language_code="en-US",
#     )

#     response = client.recognize(config=config, audio=audio)
#     transcript = response.results[0].alternatives[0].transcript

#     emit('transcript', transcript)

TELEGRAM_TOKEN='6307288419:AAGb4zkECqwCwJjBV_bV9iW4krYZ4qTNS1E'

TOKEN = TELEGRAM_TOKEN
# TOKEN2 = os.environ.get('TELEGRAM_TOKEN2')
PORT = int(os.environ.get('PORT', '8443'))
# PORT2 = int(os.environ.get('PORT2', '8444'))

# Replace 'YOUR_TELEGRAM_BOT_TOKEN' with your Telegram Bot's API token
updater = Updater(token=TOKEN, use_context=True)

# Replace 'YOUR_OPENAI_API_KEY' with your OpenAI API key
openai.api_key = os.environ.get('OPENAI_KEY')

# Replace 'YOUR_WEBHOOK_URL' with the actual webhook URL
WEBHOOK_POST_URL = "https://hook.eu1.make.com/t9wp8smpr6gm98y5pxjxf83ivvnppweq"

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def start(update: Update, context: CallbackContext) -> None:
    with app.app_context():
        user_id = update.message.from_user.id
        name = update.message.from_user.first_name
        language = update.message.from_user.language_code

        user = User(id=user_id, name=name, language=language)
        db.session.add(user)
        db.session.commit()
        
    update.message.reply_text('Hello! I am in-car concierge. Type your message starting with @voxtex_bot or send me voice in any language. I work best with complex sentences.')
    update.message.reply_text('Good example: Hey, let\'s go normal route to work, play Metallica and remind me to get groceries on the way back home. Jiz, it\'s so hot today, can you do something?')
def log_updates(update: Update, context: CallbackContext) -> None:
    logger.info(f"Incoming update: {update}")
    # Return without further processing to avoid interfering with other functions
    return  

def handle_text(update: Update, context: CallbackContext) -> None:
    # Record the start time
    start_time = time.time()
    print("Handling text message")

    if '@voxtex_bot' in update.message.text:
        # Process text message
        print("Mention of VOXTEX detected")
        
        # Remove the bot's name from the message
        message = update.message.text.replace('@maitee_bot', '')
            
        print(f"Incoming message: {message}")  # Print the incoming message
            
        # Prepare the prompt for OpenAI
        prompt = "Act as an in-car voice assistant. Provide just output as numbered list. Only if the user request contains multiple intents, divide it into separate intents, rephrase each of them to imitate commands to the car or navigation or infotainment system, and present them as a numbered list. If there is only one intent, rephrase it as a single command. User request: " + message

        # Use OpenAI's GPT-3 model to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", 
            # prompt=prompt,
            messages=[
                {
                "role": "assistant",
                "content": "Maitee"
                },
                {
                "role": "user",
                "content": prompt
                },
            ], 
            max_tokens=500,
            top_p=1, 
            temperature=0.8, 
            frequency_penalty=0.8,
            presence_penalty=0 
            # stop=["***"]
        )
        # Record the end time
        end_time = time.time()

        # Calculate the response time in milliseconds
        response_time = (end_time - start_time)
        # Process the response from OpenAI
    
        with app.app_context():
            user_id = update.message.from_user.id
            user_message = message
            bot_response = f"{response['choices'][0]['message']['content'].strip()}"
            interaction = UserInteraction(user_id=user_id, user_message=user_message, bot_response=bot_response)
            db.session.add(interaction)
            db.session.commit()

        # Send the response back to the user
        update.message.reply_text(f"{response['choices'][0]['message']['content'].strip()}\n\nTotal processing time: {response_time:.2f} seconds")

        # bullet_points = response.choices[0].message.content.strip().split("\n")
        # bullet_points_with_json = []

        # for bullet_point in bullet_points:
        #     bullet_point_text = bullet_point.strip()
        #     bullet_point_json = {
        #         "event_name": bullet_point_text,
        #         "urgency": 5  # Set the urgency level based on your logic
        #     }
        #     bullet_points_with_json.append(bullet_point_json)

        #     # Send the bullet points JSON to the webhook
        #     headers = {
        #         "Content-Type": "application/json"
        #     }
        #     data = {
        #         "bullet_points": bullet_points_with_json
        #     }
        #     response = requests.post(WEBHOOK_POST_URL, headers=headers, json=data)
            
def handle_voice(update: Update, context: CallbackContext) -> None:
    # Record the start time
    start_time = time.time()
    # Process voice message
    voice: Voice = update.message.voice
    print("Handling voice message")

    # Download the voice message
    voice_file = context.bot.get_file(voice.file_id)
    voice_file.download('voice.ogg')

    # Convert the voice message to WAV format with a sample rate of 16 kHz
    # Convert the voice message to WAV format with a sample rate of 16 kHz
    audio = AudioSegment.from_ogg('voice.ogg')
    audio = audio.set_frame_rate(16000)
    audio.export('voice.wav', format='wav')
    conversion_time = time.time() - start_time

    # Transcribe the voice message with Whisper
    response = openai.Audio.transcribe(
        model="whisper-1",
        file=open('voice.wav', 'rb'),
    )
    voice_time = time.time() - start_time - conversion_time

    print(response)
    response_str = json.dumps(response)
    # Parse the JSON string
    response_data = json.loads(response_str)
    # Process the transcription with GPT-3
    transcription = response_data['text']

    # Log the transcription
    # print(f"Transcription: {transcription}")

    # Prepare the prompt for OpenAI
    prompt = "Act as an in-car voice assistant. Provide just output as numbered list. Only if the user request contains multiple intents, divide it into separate intents, rephrase each of them to imitate commands to the car or navigation or infotainment system, and present them as a numbered list. If there is only one intent, rephrase it as a single command. User request: " + transcription

    # Use OpenAI's GPT-3 model to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k", 
        # prompt=prompt,
        messages=[
            {
            "role": "assistant",
            "content": "Maitee"
            },
            {
            "role": "user",
            "content": prompt
            },
        ], 
        max_tokens=300,
        top_p=1, 
        temperature=0.8, 
        frequency_penalty=0.8,
        presence_penalty=0 
        # stop=["***"]
    )
    print(response['choices'][0]['message']['content'].strip())
    # Record the end time
    openai_time = time.time() - start_time - conversion_time - voice_time
    end_time = time.time()
    total_time = time.time() - start_time

    # Calculate the response time in milliseconds
    response_time = (end_time - start_time)

    # print(f"OpenAI response: {response['choices'][0]['message']['content'].strip()}")  # Print the response from OpenAI
    with app.app_context():
        user_id = update.message.from_user.id
        user_message = transcription
        bot_response = f"{response['choices'][0]['message']['content'].strip()}"
        interaction = UserInteraction(user_id=user_id, user_message=user_message, bot_response=bot_response)
        db.session.add(interaction)
        db.session.commit()  

    # Send the response back to the user
    update.message.reply_text(f"{response['choices'][0]['message']['content'].strip()}\n\nTotal processing time: {response_time:.2f} seconds\nFile conversion: {conversion_time:.2f} sec\nASR: {voice_time:.2f} sec\nIntent processing: {openai_time:.2f} sec")


    # bullet_points = response.choices[0].message.content.strip().split("\n")
    # bullet_points_with_json = []

    # for bullet_point in bullet_points:
    #     bullet_point_text = bullet_point.strip()
    #     bullet_point_json = {
    #         "event_name": bullet_point_text,
    #         "urgency": 5  # Set the urgency level based on your logic
    #     }
    #     bullet_points_with_json.append(bullet_point_json)

    # # Send the bullet points JSON to the webhook
    # headers = {
    #     "Content-Type": "application/json"
    # }
    # data = {
    #     "bullet_points": bullet_points_with_json
    # }
    # try:
    #     response = requests.post(WEBHOOK_POST_URL, headers=headers, json=data)
    #     response.raise_for_status()  # Raise an exception if the request fails (status code >= 400)
    #     print("Webhook request successful")
    # except requests.exceptions.RequestException as e:
    #     print(f"Error sending webhook request: {e}")

def split_audio(file_path, segment_duration):
    audio = AudioSegment.from_file(file_path)
    total_duration = len(audio)

    segments = []
    start_time = 0
    end_time = segment_duration

    while start_time < total_duration:
        segment = audio[start_time:end_time]
        segments.append(segment)
        start_time = end_time
        end_time += segment_duration

    return segments

def handle_musicfile(update: Update, context: CallbackContext) -> None:
    # Record the start time
    start_time = time.time()
    # Process music file attachment
    music_file: Document = update.message.audio
    print("Handling audio message") 

    # Check the file extension
    file_extension = os.path.splitext(music_file.file_name)[1].lower()

    if file_extension in ['.ogg', '.mp3', '.wav']:
        # Download the music file
        music_file_path = os.path.join('music_files', str(update.message.message_id) + file_extension)
        music_file.get_file().download(music_file_path)

        # Split the music file into segments
        segment_duration = 30 * 1000  # Duration of each segment in milliseconds
        audio_segments = split_audio(music_file_path, segment_duration)

        # Transcribe each audio segment
        transcriptions = []
        for segment in audio_segments:
            # Export the audio segment as a temporary file
            segment_path = os.path.join('music_files', 'temp_segment.wav')
            segment.export(segment_path, format='wav')

            # Transcribe the audio segment with Whisper ASR API
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=open(segment_path, 'rb'),
            )
            transcription = response['text']
            transcriptions.append(transcription)

            # Remove the temporary segment file
            os.remove(segment_path)

        # Combine the transcriptions
        combined_transcription = ' '.join(transcriptions)

        # Send the transcript to the user
        update.message.reply_text(f"{combined_transcription}")

        # Prepare the prompt for OpenAI
        prompt = "Summarise the text, propose a plan to action on the ideas described. Text: " + combined_transcription

        # Use OpenAI's GPT-3 model to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", 
            # prompt=prompt,
            messages=[
                {
                "role": "assistant",
                "content": "Maitee"
                },
                {
                "role": "user",
                "content": prompt
                },
            ], 
            max_tokens=10000,
            top_p=1, 
            temperature=0.8, 
            frequency_penalty=0.8,
            presence_penalty=0 
            # stop=["***"]
        )

        # Record the end time
        end_time = time.time()
        total_time = end_time - start_time

        print(f"OpenAI response: {response.choices[0].message.content.strip()}")

        with app.app_context():
            user_id = update.message.from_user.id
            user_message = combined_transcription
            bot_response = f"{response.choices[0].message.content.strip()}"
            interaction = UserInteraction(user_id=user_id, user_message=user_message, bot_response=bot_response)
            db.session.add(interaction)
            db.session.commit()

        # Send the response back to the user
        update.message.reply_text(f"{response.choices[0].message.content.strip()}\n\nTotal processing time: {total_time:.2f} seconds")

    else:
        update.message.reply_text("Unsupported audio file format. Please send an OGG, MP3, or WAV file.")

def error_handler(update: Update, context: CallbackContext) -> None:
    logger = logging.getLogger(__name__)
    logger.warning(f"Unhandled update: {update}")

def log_updates(update: Update, context: CallbackContext) -> None:
    logger.info(f"Incoming update: {update}")

def main() -> None:
    with app.app_context():
        # Create the database
        db.create_all()
    
    # Create directories for storing audio and music files
    os.makedirs('audio_files', exist_ok=True)
    os.makedirs('music_files', exist_ok=True)

    dispatcher = updater.dispatcher

    
    # taking care of start function
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.audio, handle_musicfile))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    dispatcher.add_handler(MessageHandler(Filters.voice, handle_voice))
    dispatcher.add_error_handler(error_handler)

     # Start the webhooks
    updater.start_webhook(
        listen="0.0.0.0",
        port=int(PORT),
        url_path=TOKEN,
        webhook_url='https://talent-72e470935272.herokuapp.com/' + TOKEN
    )

    updater.idle()

if __name__ == '__main__':
    main()
    # socketio.run(app, debug=True)