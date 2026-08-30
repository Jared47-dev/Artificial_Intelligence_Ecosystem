# RAG Project

This project builds a simple retrieval-augmented generation (RAG) system using a Wikipedia article about the Golden Retriever. It reads the saved article text, splits it into smaller chunks, creates embeddings, retrieves the most relevant chunks, reranks them, and sends the best context to GPT-5 for an answer.

## Project document

The project uses the Golden Retriever Wikipedia article as its source document.

## Example questions and answers

### 1) What were Golden Retrievers originally bred for?
Answer: They were bred as gundogs—hunting companions used to retrieve game birds for sportsmen.

### 2) Where did the Golden Retriever originate?
Answer: Scotland, at Sir Dudley Marjoribanks’s Guisachan estate.

### 3) What color is a Golden Retriever's coat?
Answer: Golden

## Chunking experiment

We tested different chunking settings to see how they affected retrieval quality.

### Larger chunks: chunk_size = 1000, chunk_overlap = 200
Question: What were Golden Retrievers originally bred for?
Answer: They were bred as gundogs to retrieve shot gamebirds for hunters, especially on land (e.g., grouse and partridge).

### Smaller chunks: chunk_size = 200, chunk_overlap = 25
Question: What were Golden Retrievers originally bred for?
Answer: They were bred as gundogs for hunting—specifically to retrieve shot game (especially waterfowl) for sportsmen.

The larger chunks gave a little more surrounding detail, while the smaller chunks gave a shorter and more focused answer. The final project settings were restored to chunk_size = 500 and chunk_overlap = 50.

## Deep-dive questions

### 1. Why does the document get split into chunks before retrieval?
Because the full document is too large for one search step and too much context for a clean, focused answer. RecursiveCharacterTextSplitter breaks the article into smaller pieces that are easier to search and rank.

### 2. What do the SentenceTransformer embeddings do?
They turn each chunk and the user question into numeric vectors that represent meaning, allowing the system to compare semantic similarity.

### 3. How does FAISS retrieve the relevant chunks?
FAISS stores the chunk embeddings in an IndexFlatL2 index. The question is embedded and FAISS searches for the nearest matching vectors, which correspond to relevant text chunks.

### 4. Why does the CrossEncoder rerank the results?
The first FAISS search finds broadly relevant chunks. The CrossEncoder compares each question and chunk directly, scores them, and keeps the strongest matches before the final answer is generated.

### 5. How do chunk size and chunk overlap affect answer quality?
Chunk size controls how much context each retrieved section contains. Smaller chunks can be more focused but may lose surrounding context, while larger chunks can preserve more context but may include extra information. Overlap helps important ideas continue between neighboring chunks.

## Short project summary

The RAG flow in this project is:

read document -> split into chunks -> create embeddings -> search with FAISS -> rerank with CrossEncoder -> send best context to GPT-5.

This project keeps the design simple and beginner-friendly while still showing the core ideas behind retrieval-augmented generation.
