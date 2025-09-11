import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

dataFile=Path("playersChunkedBetter.json")
indexFile=Path("players.index")
metaFile=Path("metadata.json")

with open(dataFile, "r") as f:
    data=json.load(f)

model=SentenceTransformer("intfloat/e5-small-v2")

texts,metadata=[],[]

for i, entry in enumerate (data):
    uid=f"{entry['player_id']}_{entry['type']}_{i}"
    texts.append(entry['text'])
    metadata.append({
        "uid":uid,
        "player_id": entry["player_id"],
        "team": entry["team"],
        "type":entry["type"],
        "text":entry["text"]
    })
    print(i)
print('0')
embeddings=model.encode(texts, batch_size=16, convert_to_numpy=True)
dim=embeddings.shape[1]
print('1')
index=faiss.IndexFlatL2(dim)
index.add(embeddings)
print('2')
faiss.write_index(index, str(indexFile))
print('3')
with open(metaFile, 'w') as f:
    json.dump(metadata, f, indent=2)

print('done')

query="is patrick mahome good"
query_embedding=model.encode([query])

k=2
distances, indices=index.search(query_embedding,k)

print(f"Query{query}")

for i, idx in enumerate(indices[0]):
    print(f"{texts[idx]} (Distance: {distances[0][i]})")