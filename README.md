# Interactive Voice Response

This repo consists of an audio folder, a jupyter notebook, and a python file.

The core of this repo is an Interactive Voice Response (IVR) app that plays audio, records reply, identifies intent, and replies in response to the intent.

There are 2 ways to interact with the app:
- Jupyter notebook: Interactive voice response
- Use command line: python interactive_voice_response.py

Models used in the app: 
- OpenAI's whisper for speech recognition
- ollama's Gemma:2B for few-shot and zero-shot learning to identify intent

The IVR app scenario is the Australian Tax Office phone line, looking to classify the customer's inquiry into 1 of 5 intents:
- Check the progress of your tax return
- Request your existing tax file number
- Tax return preparation
- Linking myGov to myTax

Phrases you can use to try the app:
- When do I get my tax return? Tax return progress?
- What is my tax file number? What is my TFN?
- I need help preparing my tax return. How I can do my tax return?
- I need to link myGov to myTax. How can I link myGov account to myTax?

