# Import the libraries
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import ollama
import json
import logging
from transformers import pipeline

# Set the file name and format of the log file
log_file = "transcript.log"
log_format = "%(asctime)s %(message)s"

# Configure the logging settings
logging.basicConfig(filename=log_file, format=log_format, level=logging.INFO)

# Load the pre-trained model for intent recognition
model = whisper.load_model("base").to('cpu')
classifier = pipeline(task="sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')

#This is function to play audio

def playaudio(file):
    
    # Read the audio file as a NumPy array
    data, fs = sf.read(file, dtype='float32')

    # Play the audio file using sounddevice
    sd.play(data, fs)

    # Wait until the file is done playing
    status = sd.wait()
    
    return data

#This is function to record audio

def record_audio(file, duration):
    # Set the sampling frequency
    fs = 44100
    # Record audio using sounddevice
    data = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    print("Recording started...")
    # Wait until the recording is done
    status = sd.wait()
    print("Recording stopped.")
    # Save the recorded audio to a file using soundfile
    sf.write(file, data, fs)

def confirm_message(intent):
    if intent == 'Check the progress of your tax return':
        playaudio('audio/check.wav')
    elif intent == 'Request your existing tax file number':
        playaudio('audio/request.wav')
    elif intent == 'Tax return preparation':
        playaudio('audio/prepare.wav')
    elif intent == 'Linking myGov to myTax':
        playaudio('audio/link.wav')
    else:
        print('confused')


# Main program
if __name__ == "__main__":
    
    print("Please have audio on.")
    # Play an audio file to greet the user and ask for their query
    playaudio('audio/greetings.wav')
    #This is to record the customer's intent
    print("Say your intent in 5 seconds.")
    record_audio('audio/intent.wav', 5)
    #This is to transcribe the customer's intent.
    intent = model.transcribe('audio/intent.wav')

    #Other than the one the customer records, these are all the 4 intents I pre-recorded, which can help if you want to test different intents

    #intent = model.transcribe("audio/check_question.wav")
    #intent = model.transcribe("audio/prepare_question.wav")
    #intent = model.transcribe("audio/request_question.wav")
    #intent = model.transcribe("audio/link_question.wav")
    
    print('I heard you said: '+intent['text'])

    # Write a message to the log file
    transcription_intent = intent['text']
    logging.info(transcription_intent)

    #This is to identify the sentiment of the customer's intent
    sentiment = classifier(transcription_intent)
    print(sentiment)

    #This is to print operator, if the intent is negative enough
    if sentiment[0]['label'] == 'NEGATIVE' and sentiment[0]['score'] > 0.99:
        print("operator")
    else: 
        print('Sounds good. Let me direct you to the right route.')

    #This is to use few-shot learning to classify the intent of the customer into 1 of 4 pre-designed intents
    intent_prompt = 'There are 5 types of intents: Check the progress of your tax return, Request your existing tax file number, Tax return preparation, Linking myGov to myTax, and Irrelevant. If the text is like "When do I get my tax return?" or "Tax return progress", the intent is "Check the progress of your tax return". If the text is like "What is my tax file number?" or "What is my TFN?", the intent is "Request your existing tax file number". If the text is like "I need help preparing my tax return" or "How I can do my tax return?", the intent is "Tax return preparation". If the text is like "I need to link my myGov to myTax" or "How can I link myGov to myTax", the intent is "Linking myGov to myTax". Otherwise, the text is "Irrrelevant". Tag the following text with one of the intents: How can I do my tax return'

    r = ollama.generate(
        model='gemma:2b', 
        prompt=intent_prompt,
        format='json')

    answers = json.loads(r['response'])
    print('I figure your intent is: '+answers['intent'])

    #This is to play a confirmation question audio following the classification of customer's intent
    confirm_message(answers['intent'])

    #This is to record the customer's response to the confirmation question
    print("Say your response in 3 seconds.")
    record_audio('audio/response.wav', 3)
    
    #This is to transcribe the response
    response = model.transcribe('audio/response.wav')
    transcription_response = response['text']
    print('I heard you said: '+transcription_response)

    #This is to classify the response as affirmative or negative
    response = ollama.generate(model='gemma:2b', prompt='Classify the following as saying yes or no: '+transcription_response)

    print(response['response'])

    #This is to play the final thanking audio, and print out the intent that was identified.

    for word in ['yes', 'yep', 'yeah', 'true', 'yea']:
        if word in response['response'].lower():
            playaudio('audio/thanks.wav')
            print(answers['intent'])
            break
    else:
        print('confused')

   
    
    
