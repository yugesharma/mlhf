import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from dotenv import load_dotenv
import app

# load_dotenv()

class Token:
    def __init__(self, token): self.token = token

def test_api_requires_token():
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "HF_TOKEN not set in environment"

    gen = app.chatBot(
        hf_token=Token(None),
        query="Who is Patrick Mahomes",
        max_tokens=8,
        temperature=0.2,
        top_p=0.9,
        use_local_model=True,
    )
    
    assert "please log in" not in gen.lower()  # shouldn't get warning
    assert isinstance(gen, str)
    assert len(gen)>0
