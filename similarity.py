import sacrebleu
import re
from typing import List


st_model = None
def get_similarity_batched(texts1: List[str], texts2: List[str]):
    import torch
    from sentence_transformers import SentenceTransformer, util
    global st_model
    if st_model is None:
        #paraphrase-multilingual-mpnet-base-v2
        #all-MiniLM-L12-v2
        #all-distilroberta-v1
        #all-mpnet-base-v2
        #all-MiniLM-L6-v2
        st_model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu', cache_folder="./s_cache")
    
    clean_text_batch(texts1, texts2)
    embeddings1 = st_model.encode(texts1, convert_to_tensor=True, show_progress_bar=False)
    embeddings2 = st_model.encode(texts2, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.diag().tolist()

def clean_text_batch(texts1: List[str], texts2: List[str]): 
    if len(texts1) == len(texts2):
        for i in range(0, len(texts1)):
            texts1[i] = clean_text(texts1[i], stricter= True)
            texts2[i] = clean_text(texts2[i], stricter= True)
    #


def clean_text(text, stricter=False):
    if stricter:
        text = re.sub(r"([^a-zA-Z]|^)([a-zA-Z])(?i:-\2)+([a-zA-Z])", r"\1\2\3", text)
    to_strip = "&っ。～―（）「」｢｣『』“”\"'，、○()«»~ \t\r\n"
    if stricter:
        to_strip += "….?？!！,"
    text = text.strip(to_strip)
    return text

def get_similarity(ref, hyp):
    ref = clean_text(ref, stricter=True)
    if not ref:
        return 1.0
    hyp = clean_text(hyp, stricter=True)
    if ref.lower() == hyp.lower():
        return 1.0
    return float(get_similarity_batched([ref], [hyp])[0])

def get_bleu(ref, hyp):
    ref = clean_text(ref)
    hyp = clean_text(hyp)
    if ref.lower() == hyp.lower():
        return 100
    bleu = sacrebleu.sentence_bleu(hyp, [ref])
    return bleu.score

def get_chrf(ref, hyp):
    ref = clean_text(ref)
    hyp = clean_text(hyp)
    if ref.lower() == hyp.lower():
        return 100
    chrf = sacrebleu.sentence_chrf(hyp, [ref])
    return chrf.score

