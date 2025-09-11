'''
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
'''

import json
from locale import normalize
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

db=faiss.read_index("database/players.index")
with open("database/metadata.json") as f:
    metadata=json.load(f)

model=SentenceTransformer("intfloat/e5-small-v2")

print("NFL player RAG search")
print("Type 'end' to quit")

while True:
    query=input("Ask a question: ")
    if query.lower()=='end':
        break

    qEmbed=model.encode([f"query: {query}"], normalize_embeddings=True)

    k=5
    D, I=db.search(np.array(qEmbed), k)

    for rank, idx in enumerate(I[0]):
        item=metadata[idx]
        print(f"\n[{rank}] Team: {item['team']} | Type: {item['type']} | Score: {D[0][rank]: .3f}")
        print(f"Text: {item['text']}")