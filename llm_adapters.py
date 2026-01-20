import asyncio
import re
from typing import Optional

import requests
from langchain_openai import ChatOpenAI

from conf import settings
from rag_lib import get_llm


class AsyncOllamaChatLLM:
    """Async chat client for Ollama HTTP API providing a uniform chat(prompt) -> str surface."""

    def __init__(self, model: str = 'deepseek-r1:32b', base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url

    async def chat(self, prompt: str) -> str:
        try:
            url = f"{self.base_url}/api/chat"
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0},
            }
            resp = await asyncio.to_thread(requests.post, url, json=data)
            resp.raise_for_status()
            result = resp.json()
            content = result.get('message', {}).get('content', '')
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return content.strip()
        except Exception as exc:
            print(f"[AsyncOllamaChatLLM] Error calling Ollama: {exc}")
            return ""


class AsyncOpenAIChatAdapter:
    """Adapter that exposes async chat(prompt) on sync ChatOpenAI-like clients."""

    def __init__(self, chat_llm):
        self.chat_llm = chat_llm

    async def chat(self, prompt: str) -> str:
        try:
            resp = await asyncio.to_thread(self.chat_llm.invoke, prompt)
            return resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
        except Exception as exc:
            print(f"[AsyncOpenAIChatAdapter] Error invoking LLM: {exc}")
            return ""


def get_async_chat_llm(config_section: Optional[str] = 'knowledge_graph_llm'):
    """Return an async chat-capable LLM client configured by the given settings section."""
    llm_conf = None
    if config_section and config_section in settings.config:
        llm_conf = settings.config[config_section]

    if not llm_conf:
        return AsyncOpenAIChatAdapter(get_llm())

    llm_type = llm_conf.get('type', 'ollama').lower()
    if llm_type == 'ollama':
        return AsyncOllamaChatLLM(
            model=llm_conf.get('model', 'deepseek-r1:32b'),
            base_url=llm_conf.get('base_url', 'http://localhost:11434'),
        )

    if llm_type == 'openai':
        chat = ChatOpenAI(
            model=llm_conf.get('model'),
            api_key=llm_conf.get('api_key'),
            base_url=llm_conf.get('base_url'),
            temperature=float(llm_conf.get('temperature', 0.2)),
        )
        return AsyncOpenAIChatAdapter(chat)

    return AsyncOpenAIChatAdapter(get_llm())
