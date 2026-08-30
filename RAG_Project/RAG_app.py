import logging
import warnings

import openai
from transformers import logging as hf_logging
from dotenv import load_dotenv
import os

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8

with open("Selected_Document.txt", "r", encoding="utf-8") as file:
    text = file.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""],
)

chunks = text_splitter.split_text(text)

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=False)
embedding_array = np.array(embeddings, dtype="float32")
embedding_dim = embedding_array.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(embedding_array)


def retrieve_chunks(question, k=top_k):
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.array(q_vec, dtype="float32")
    distances, I = faiss_index.search(q_arr, k)
    return [chunks[i] for i in I[0]]


from sentence_transformers import CrossEncoder

reranker = CrossEncoder(cross_encoder_name)


def dedupe_preserve_order(items):
    seen = set()
    cleaned = []
    for item in items:
        normalized = " ".join(item.split())
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def rerank_chunks(question: str, candidate_chunks: list[str], m: int = top_m) -> list[str]:
    pairs = [(question, chunk) for chunk in candidate_chunks]
    scores = reranker.predict(pairs)

    ranked = [chunk for _, chunk in sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)]
    top_chunks = ranked[:m]
    return dedupe_preserve_order(top_chunks)


def answer_question(question):
    candidates = retrieve_chunks(question)
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)
    context = "\n\n".join(relevant_chunks)

    system_prompt = "You are a knowledgeable assistant that answers questions based on the provided context. If the answer is not in the context, say you don’t know."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    resp = openai.chat.completions.create(
        model="gpt-5",
        max_completion_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))
