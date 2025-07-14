from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from similarity import get_similarity_batched, get_bleu, get_chrf


app = FastAPI(
    title="Sentence similarity API", 
    description="Check Sentences similarities.", 
    version="1.0"
)


class Texts(BaseModel):
    texts1: List[str]
    texts2: List[str]

@app.get("/")
def home():
    #return {"mensagem": "Bem-vindo à API!"}
    return RedirectResponse(url="/docs")


@app.post('/api/similarity')
def get_sim(texts: Texts): 
    result = []
    sim = get_similarity_batched(texts.texts1, texts.texts2)
    for i in range(0, len(texts.texts1)):
        result.append({
            "bleu": get_bleu(texts.texts1[i], texts.texts2[i]),
            "chrf": get_chrf(texts.texts1[i], texts.texts2[i]),
            "similarity": sim[i]
        })

    return result