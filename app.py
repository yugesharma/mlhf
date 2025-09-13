import gradio as gr # type: ignore
from huggingface_hub import InferenceClient
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv
load_dotenv()


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

nfl_css = """
@import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;600;700&display=swap');

:root{
  --nfl-blue:#013369;
  --nfl-red:#D50A0A;
  --nfl-silver:#A5ACAF;
  --turf:#0b6623;
  --yard:rgba(255,255,255,0.12);
}

#main-container{
  font-family:'Teko',sans-serif;
  background:
    linear-gradient(90deg,transparent 0 97%,var(--yard) 97% 100%),
    repeating-linear-gradient(0deg,var(--yard) 0 2px,transparent 2px 60px),
    radial-gradient(circle at 20% 10%,rgba(255,255,255,0.05),transparent 40%),
    linear-gradient(180deg,rgba(0,0,0,0.35),rgba(0,0,0,0.35)),
    linear-gradient(180deg,#0b6623,#0a5a1f);
  min-height:100vh;
  background-size:cover;
  background-attachment:fixed;
  color:#fff;
}

.gradio-container{
  max-width:860px;
  margin:24px auto;
  padding:24px;
  background:rgba(1,51,105,0.85);
  box-shadow:0 12px 28px rgba(0,0,0,0.6);
  border-radius:16px;
  border:2px solid var(--nfl-silver);
  color:#fff;
}

#title h1{
  text-align:center;
  font-size:3.2em;
  letter-spacing:1px;
  margin:8px 0 20px;
  color:#fff;
  text-shadow:0 3px 0 var(--nfl-red),0 6px 12px rgba(0,0,0,0.6);
}

.gr-button{
  background:linear-gradient(180deg,var(--nfl-red),#9d0808);
  color:#fff;
  border:2px solid var(--nfl-silver);
  border-radius:10px;
  padding:12px 22px;
  cursor:pointer;
  text-transform:uppercase;
  letter-spacing:.8px;
  transition:transform .15s ease,box-shadow .15s ease,filter .15s ease;
}

.gr-button:hover{
  transform:translateY(-1px) scale(1.03);
  box-shadow:0 6px 16px rgba(213,10,10,0.4);
  filter:saturate(1.1);
}

.gr-button:active{
  transform:translateY(0) scale(0.99);
}

.gr-input,.gr-textarea,.gr-text,.gr-output-text,textarea,input,select{
  background-color:rgba(165,172,175,0.15) !important;
  color:#fff !important;
  border:1px solid var(--nfl-silver) !important;
  padding:12px 14px;
  border-radius:8px;
  font-size:1.05em;
}

.gr-output-text{
  background-color:rgba(1,51,105,0.35) !important;
  border:2px solid var(--nfl-red) !important;
}

label,.gr-label{
  color:#e6edf2 !important;
  font-weight:600;
  font-size:1.05em;
  margin-bottom:6px;
  letter-spacing:.5px;
  text-transform:uppercase;
}

.gr-slider>label{
  color:#fff !important;
}

.gr-slider>div>div>input[type="range"]{
  accent-color:var(--nfl-red);
}

.gr-slider>div>div>span{
  color:#fff !important;
}

.gradio-container a,a{
  color:#ffeb99;
  text-decoration:none;
}

.gradio-container a:hover,a:hover{
  text-decoration:underline;
}
"""


def chatBot(query,  
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
    use_local_model: bool):


    API_MODEL_NAME = "Qwen/Qwen2-7B-Instruct" 
    print("Loading RAG database and retriever model...")
    db = faiss.read_index("database/players.index")
    with open("database/metadata.json") as f:
        metadata = json.load(f)
    retriever_model = SentenceTransformer("intfloat/e5-small-v2")
    print("RAG components loaded.")

    if use_local_model:
        print("[MODE] Using local model.")
        llmName = "Qwen/Qwen3-0.6B"
        try:
            tokenizer = AutoTokenizer.from_pretrained(llmName)
            llm = AutoModelForCausalLM.from_pretrained(
                llmName,
                torch_dtype="auto",
                device_map="auto"
            )
            print(f"Local model '{llmName}' loaded successfully.")
        except Exception as e:
            print(f"Error loading local model: {e}")
            llm, tokenizer = None, None
    else:
        print(f"[MODE] Using Hugging Face API for model: openai/gpt-oss-20b")
        if hf_token is None or not getattr(hf_token, "token", None):
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        client = InferenceClient(model="openai/gpt-oss-20b", token=hf_token.token)
        print("API client initialized.")
    qEmbed = retriever_model.encode([f"query: {query}"], normalize_embeddings=True)
    k = 3
    score, index = db.search(np.array(qEmbed), k)

    contextList = [metadata[idx]['text'] for idx in index[0]]
    context = "\n".join(contextList)
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

    if use_local_model:
        if not llm or not tokenizer:
            return "Local model is not loaded. Please check for errors at startup."
            
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)
        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content
    
    else:
        try:
            response = client.chat_completion(
                messages,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling API: {e}"


demo = gr.Interface(
    fn=chatBot,
    inputs=[
        gr.Textbox(label="Ask about a player...", placeholder="e.g., Who is Deshaun Watson?", lines=2),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max New Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-P (Nucleus Sampling)"),
        gr.Checkbox(label="Use Local Model", value=False, info="Uncheck to use Hugging Face Inference API.")
    ],
    outputs=[gr.Textbox(label="NFL Bot's Answer", lines=10, max_lines=30)],
)

with gr.Blocks(css=nfl_css, elem_id="main-container") as chatbot:
    with gr.Row():
        gr.Markdown("<div id='title'><h1>NFL News Bot 🏈</h1></div>")
        gr.LoginButton()
    demo.render()

if __name__ == "__main__":
    chatbot.launch()

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