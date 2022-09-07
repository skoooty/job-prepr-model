from transformers import pipeline
from fastapi import FastAPI, Request

app = FastAPI()
app.state.model = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')

@app.get("/hello")
def hello():
    return 'hello'


@app.get("/predictnlp")
async def predict_nlp(text):
    print(app.state.model(text))
    return app.state.model(text)
