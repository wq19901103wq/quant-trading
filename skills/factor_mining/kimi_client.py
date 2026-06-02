"""Kimi (Moonshot) LLM Client."""
import os
import json
from typing import Optional
import requests
from skills.factor_mining.skill import LLMClient


class KimiLLMClient(LLMClient):
    """使用 Kimi API 的客户端（直接 HTTP 调用，无需 openai 包）."""

    BASE_URL = "https://api.moonshot.cn/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: str = "moonshot-v1-8k"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Kimi API key not found. Set OPENAI_API_KEY env var.")
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 800,
        }
        response = requests.post(
            self.BASE_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
