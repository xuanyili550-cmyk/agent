from transformers import AutoTokenizer,AutoModel
import torch

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer=AutoTokenizer.from_pretrained(model_ckpt)
model=AutoModel.from_pretrained(model_ckpt)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]

def get_embeddings(text_list):
    enc=tokenizer(text_list,padding=True,truncation=True,return_tensors='pt')
    enc={k:v.to(device) for v ,k in enc.items()}
    return cls_pooling(model(**enc))


EMBED_CKPT = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
CORPUS = [
    "You can load a dataset offline by using a local cache without internet.",
    "To fine-tune a model, use the Trainer API together with TrainingArguments.",
    "FAISS builds an index that lets you quickly find similar embeddings.",
    "Tokenizers convert text into input_ids that the model can process.",
    "Padding makes all sequences in a batch the same length.",
    "The attention mask tells the model which tokens to ignore.",
    "Semantic search finds documents by meaning, not by exact keywords.",
    "Streaming lets you use a dataset that is larger than your memory.",
    "Mean pooling averages token vectors to produce a sentence embedding.",
    "You can deploy a model as an OpenAI-compatible API server.",
]



def load_embedder():
    tok = AutoTokenizer.from_pretrained(EMBED_CKPT)
    model = AutoModel.from_pretrained(EMBED_CKPT).to(device)
    model.eval()
    def get_embeddings(text_list):
        enc=tokenizer(text_list,padding=True,truncation=True,return_tensors='pt')
        enc={k:v.to(device) for v,k in enc.items()}
        with torch.no_grad():
            out=model(**enc)
        return out.last_hidden_state[:,0]
    return tok,model,get_embeddings

from datasets import Dataset

ds = Dataset.from_dict({
    "review": ["great drug, helped a lot", "no effect at all", "", "side effects were bad"],
    "rating": [9, 3, 5, 2],
})
ds = ds.filter(lambda x: len(x["review"]) > 0)
ds = ds.map(lambda x: {"n_words": len(x["review"].split())})
ds = ds.rename_column("rating", "score")

tok, model, get_embeddings = load_embedder()

text = "How can I load a dataset offline?"
emb = get_embeddings([text])
print(tuple(emb.shape))
corpus_emb = get_embeddings(CORPUS)
query = "How do I use a dataset without internet?"
q_emb = get_embeddings([query])
scores = (q_emb @ corpus_emb.T)[0]
topk = torch.topk(scores, 3)
ds = Dataset.from_dict({"text": CORPUS})
ds = ds.map(lambda x: {"embeddings": get_embeddings([x["text"]]).cpu().numpy()[0]})
ds.add_faiss_index(column="embeddings")
query = "way to search text by meaning"
q_emb = get_embeddings([query]).cpu().numpy()
scores, samples = ds.get_nearest_examples("embeddings", q_emb, k=3)

corpus_emb = get_embeddings(CORPUS)
query = "process data that does not fit in RAM"
q_words = set(query.lower().split())
kw_hits = [d for d in CORPUS if q_words & set(d.lower().split())]
scores = (get_embeddings([query]) @ corpus_emb.T)[0]
best = CORPUS[torch.argmax(scores).item()]
ds = Dataset.from_dict({"text": CORPUS})
ds = ds.map(lambda x: {"embeddings": get_embeddings([x["text"]]).cpu().numpy()[0]})
ds.add_faiss_index(column="embeddings")

def search(query, k=2):
    q_emb = get_embeddings([query]).cpu().numpy()
    scores, samples = ds.get_nearest_examples("embeddings", q_emb, k=k)
    return list(zip(scores, samples["text"]))
questions = [
        "how to train a model on my own data",
        "make batches the same length",
        "serve a model behind an API",
    ]