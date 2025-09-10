import gradio as gr
from huggingface_hub import InferenceClient
import os
import requests
from datetime import datetime

SPORTS_IO_API_KEY = os.getenv("6f9c8d8609a744ce88440fec15e711b7")
NEWS_URL = "https://api.sportsdata.io/v3/nfl/scores/json/News?"
PLAYERS_URL = "https://api.sportsdata.io/v3/nfl/scores/json/Players?"
PLAYER_STATS_URL = "https://api.sportsdata.io/v3/nfl/stats/json/PlayerSeasonStatsByPlayerID/{season}/{playerid}?"

CURRENT_SEASON = 2025

#headers = {"Ocp-Apim-Subscription-Key": SPORTS_IO_API_KEY}

# Fancy styling
fancy_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #45a049;
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""

def fetch_news(query: str):
    try:
        resp = requests.get(NEWS_URL, key=SPORTS_IO_API_KEY, timeout=10)
        resp.raise_for_status()
        articles = resp.json()
    except Exception as e:
        return f"⚠️ News API error: {e}"

    results = []
    for article in articles:
        if query.lower() in (article.get("Title", "") + article.get("Content", "")).lower():
            date = datetime.fromisoformat(article["Updated"].replace("Z", "+00:00"))
            results.append({
                "title": article["Title"],
                "content": article["Content"],
                "source": article["Source"],
                "updated": date.strftime("%B %d, %Y %I:%M %p")
            })

    if not results:
        return f"No news found for '{query}'."

    results = sorted(results, key=lambda x: x["updated"], reverse=True)[:3]
    formatted = "### 📰 Latest News\n"
    for r in results:
        formatted += f"**{r['title']}**  \n{r['content'][:250]}...  \n"
        formatted += f"_Source: {r['source']} • {r['updated']}_\n\n"
    return formatted

def fetch_player_stats(player_name: str):
    try:
        resp = requests.get(PLAYERS_URL, key=SPORTS_IO_API_KEY, timeout=15)
        resp.raise_for_status()
        players = resp.json()
    except Exception as e:
        return f"⚠️ Player lookup error: {e}"

    # Find player by name
    player = next((p for p in players if player_name.lower() in p["Name"].lower()), None)
    if not player:
        return None  # Not a player

    player_id = player["PlayerID"]
    team = player.get("Team", "Unknown")
    position = player.get("Position", "N/A")

    # Fetch season stats
    try:
        stats_url = PLAYER_STATS_URL.format(season=CURRENT_SEASON, playerid=player_id)
        resp = requests.get(stats_url, key=SPORTS_IO_API_KEY, timeout=10)
        resp.raise_for_status()
        stats = resp.json()
    except Exception as e:
        return f"⚠️ Stats fetch error: {e}"

    if not stats:
        return f"📊 No stats available yet for {player_name} ({CURRENT_SEASON})."

    formatted = f"### 📊 {player_name} ({team}, {position}) — {CURRENT_SEASON} Season Stats\n"
    if position in ["QB", "Quarterback"]:
        formatted += f"- Passing Yards: {stats.get('PassingYards', 0)}\n"
        formatted += f"- Passing TDs: {stats.get('PassingTouchdowns', 0)}\n"
        formatted += f"- Interceptions: {stats.get('PassingInterceptions', 0)}\n"
    elif position in ["RB", "Running Back"]:
        formatted += f"- Rushing Yards: {stats.get('RushingYards', 0)}\n"
        formatted += f"- Rushing TDs: {stats.get('RushingTouchdowns', 0)}\n"
    elif position in ["WR", "TE"]:
        formatted += f"- Receiving Yards: {stats.get('ReceivingYards', 0)}\n"
        formatted += f"- Receiving TDs: {stats.get('ReceivingTouchdowns', 0)}\n"
        formatted += f"- Receptions: {stats.get('Receptions', 0)}\n"
    else:
        formatted += f"- Games Played: {stats.get('Played', 0)}\n"
        formatted += f"- Fantasy Points: {stats.get('FantasyPoints', 0)}\n"

    return formatted

def nfl_chatbot(query: str, history: list):
    news = fetch_news(query)
    stats = fetch_player_stats(query)

    if stats is None:
        reply = f"{news}\n\n(ℹ️ No player stats — likely a coach.)"
    else:
        reply = f"{news}\n\n{stats}"

    history.append((query, reply))
    return history, ""

with gr.Blocks(css=fancy_css) as demo:
    gr.Markdown("# 🏈 NFL News + Stats Chatbot")
    gr.Markdown("Ask about any NFL player or coach for the **latest news and stats**.")

    chatbot = gr.Chatbot()
    query = gr.Textbox(label="Enter player or coach name")
    btn = gr.Button("Get Update")

    btn.click(nfl_chatbot, [query, chatbot], [chatbot, query])
    query.submit(nfl_chatbot, [query, chatbot], [chatbot, query])

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