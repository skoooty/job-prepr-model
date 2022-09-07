from google.cloud import speech_v1 as speech
from fastapi import FastAPI, UploadFile
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, Trainer
import io
import os
import json
from pydub import AudioSegment

app = FastAPI()
app.state.tokenizer, app.state.model = load_nlp()

def transcribe(source):
    """Transcribe the given audio file from a local or bucket path"""
    if os.environ.get("TRANSCRIPTION_SOURCE") == "local":
        client = speech.SpeechClient()

        with io.open(source, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        audio_channel_count=1
        )
    else:
        config = dict(language_code="en-UK", enable_automatic_punctuation=True)
        audio = dict(uri=source)

    client = speech.SpeechClient()
    response = client.recognize(config=config, audio=audio)
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
