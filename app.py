import gradio as gr
from huggingface_hub import InferenceClient
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

db=faiss.read_index("database/players.index")
with open("database/metadata.json") as f:
    metadata=json.load(f)

model=SentenceTransformer("intfloat/e5-small-v2")

llmName="Qwen/Qwen3-0.6B"
tokenizer=AutoTokenizer.from_pretrained(llmName)
llm=AutoModelForCausalLM.from_pretrained(
    llmName,
    torch_dtype="auto",
    device_map="auto"
)

def chatBot(query):
    db=faiss.read_index("database/players.index")
    with open("database/metadata.json") as f:
        metadata=json.load(f)

    model=SentenceTransformer("intfloat/e5-small-v2")

    llmName="Qwen/Qwen3-0.6B"
    tokenizer=AutoTokenizer.from_pretrained(llmName)
    llm=AutoModelForCausalLM.from_pretrained(
        llmName,
        torch_dtype="auto",
        device_map="auto"
    )
    # query=input("\nEnter your question: \n")
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
    return content

demo=gr.Interface(
    fn=chatBot,
    inputs=["text"],
    outputs=["text"]
)

if __name__ == "__main__":
    demo.launch()

'''
pipe = None
stop_inference = False



def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
    use_local_model: bool,
):
    global pipe

    # Build messages from history
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""

    if use_local_model:
        print("[MODE] local")
        from transformers import pipeline
        import torch
        if pipe is None:
            pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

        # Build prompt as plain text
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        response = outputs[0]["generated_text"][len(prompt):]
        yield response.strip()

    else:
        print("[MODE] api")

        if hf_token is None or not getattr(hf_token, "token", None):
            yield "⚠️ Please log in with your Hugging Face account first."
            return

        client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content
            response += token
            yield response


chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(label="Use Local Model", value=False),
    ],
    type="messages",
)

with gr.Blocks(css=fancy_css) as demo:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'>🌟 Fancy AI Chatbot 🌟</h1>")
        gr.LoginButton()
    chatbot.render()

if __name__ == "__main__":
    demo.launch()
'''