import requests
import os

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        
    def get_completion(self, model: str, messages: list[dict], max_new_tokens: int = 256) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model, 
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            },
        )
        return response.json()["choices"][0]["message"]["content"]