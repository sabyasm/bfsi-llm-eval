"""Unified LLM client supporting OpenAI, Anthropic, Mistral, and custom endpoints."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
logger = logging.getLogger(__name__)

RAW_RESPONSE_DIR = Path("data/generated/raw_llm_responses")


class LLMClient:
    """Unified interface for calling generation LLMs."""

    def __init__(self, config: dict):
        self.provider: str = config["provider"]
        self.model_name: str = os.environ.get("MODEL_NAME", config["model_name"])
        self.temperature: float = config.get("temperature", 0.7)
        self.max_tokens: int = config.get("max_tokens", 1500)
        self._api_key = os.environ.get(config.get("api_key_env", "GENERATION_API_KEY"), "")
        self._client = None
        self._init_client()
        RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

    def _init_client(self) -> None:
        if self.provider == "openai" or self.provider == "custom":
            import openai
            kwargs = {"api_key": self._api_key}
            if self.provider == "custom":
                kwargs["base_url"] = os.environ.get("API_BASE", os.environ.get("CUSTOM_LLM_BASE_URL", ""))
            self._client = openai.OpenAI(**kwargs)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        elif self.provider == "mistral":
            from mistralai import Mistral
            self._client = Mistral(api_key=self._api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the LLM. Returns raw text."""
        if self.provider in ("openai", "custom"):
            return self._generate_openai(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt)
        elif self.provider == "mistral":
            return self._generate_mistral(prompt, system_prompt)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_openai(self, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or ""
        self._log_raw(prompt, text)
        return text

    def _generate_anthropic(self, prompt: str, system_prompt: str) -> str:
        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        self._log_raw(prompt, text)
        return text

    def _generate_mistral(self, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or ""
        self._log_raw(prompt, text)
        return text

    def _log_raw(self, prompt: str, response: str) -> None:
        """Append raw LLM exchange to log file."""
        log_path = RAW_RESPONSE_DIR / "responses.jsonl"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "model": self.model_name,
                    "prompt_preview": prompt[:200],
                    "response_preview": response[:500],
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
