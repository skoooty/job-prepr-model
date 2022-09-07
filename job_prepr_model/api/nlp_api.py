from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from fastapi import FastAPI, UploadFile
from api.transcription import transcribe

app = FastAPI()
app.state.tokenizer, app.state.model = load_nlp()

def load_nlp():
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

@app.post("/predict-nlp")
async def predict_nlp(audio : UploadFile):
    with io.open(audio, "rb") as audio_file:
            content = audio_file.read()
    pred_text = transcribe(content)
    tokenized_texts = app.state.tokenizer(pred_text,truncation=True,padding=True)
    print(app.state.model.predict(tokenized_texts))
