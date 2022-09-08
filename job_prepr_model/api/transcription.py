<<<<<<< HEAD
from google.cloud import speech_v1p1beta1 as speech
import io
=======
from google.cloud import speech_v1 as speech
from fastapi import FastAPI, UploadFile
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, Trainer
import io
import os
import json
from pydub import AudioSegment

app = FastAPI()
app.state.tokenizer, app.state.model = load_nlp()
>>>>>>> master

def transcribe(source):
    """Transcribe the given audio file from a local or bucket path"""
    with io.open(source, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.MP3,
    sample_rate_hertz=48000,
    language_code="en-US",
    audio_channel_count=1
    )
    client = speech.SpeechClient()
    response = client.recognize(config=config, audio=audio)
    best_alternative = speech.SpeechRecognitionAlternative()
    for result in response.results:
        best_alternative = result.alternatives[0]
    transcript = best_alternative.transcript
    return transcript

def load_nlp():
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def mp3_to_wav(audio):
    sound = AudioSegment.from_mp3(audio)
    sound.export(dst, format="wav")

@app.post("/predict-nlp")
async def predict_nlp(audio : UploadFile):
    pred_text = transcribe(audio)
    tokenized_texts = app.state.tokenizer(pred_text,truncation=True,padding=True)
    print(app.state.model.predict(tokenized_texts))
