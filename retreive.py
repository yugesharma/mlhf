

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

db=faiss.read_index("database/players.index")
with open("database/metadata.json") as f:
    metadata=json.load(f)

model=SentenceTransformer("intfloat/e5-small-v2")

print("NFL player RAG search")
print("Type 'end' to quit")

# while True:
#     query=input("Ask a question: ")
#     if query.lower()=='end':
#         break

#     qEmbed=model.encode([f"query: {query}"], normalize_embeddings=True)

#     k=3
#     score, index=db.search(np.array(qEmbed), k)
#     # print(index, dis)

#     context=[]
#     for rank, idx in enumerate(index[0]):
#         item=metadata[idx]
#         context.append
#         print(f"\n[{rank}] Team: {item['team']} | Type: {item['type']} | Score: {score[0][rank]}")
#         print(f"Text: {item['text']}")


llmName="Qwen/Qwen3-0.6B"
tokenizer=AutoTokenizer.from_pretrained(llmName)
llm=AutoModelForCausalLM.from_pretrained(
    llmName,
    torch_dtype="auto",
    device_map="auto"
)



query=input("\nEnter your question: \n")
    


qEmbed=model.encode([f"query: {query}"], normalize_embeddings=True)

k=3
score, index=db.search(np.array(qEmbed), k)
# print(index, dis)

contextList=[]
for rank, idx in enumerate(index[0]):
    item=metadata[idx]
    # print(item)
    contextList.append(item['text'])
context="\n".join(contextList)
    # print(f"\n[{rank}] Team: {item['team']} | Type: {item['type']} | Score: {score[0][rank]}")
    # print(f"Text: {item['text']}")

prompt = f"""
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {query}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""


messages = [
    {"role": "user", "content": prompt}
]



#For results with thinking content, comment out below code and comment the code under 'without thinking content'

text=tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

model_inputs=tokenizer([text], return_tensors="pt").to(llm.device)
generated_ids = llm.generate(
    **model_inputs,
    max_new_tokens=1024
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
#parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n") 
print("thinking content: ", thinking_content )



print("content: ", content)

'''
==Good queries to try==
Tell me about Will Johnson
Who is quarterback for the arizona cardinals?
'''