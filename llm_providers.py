import os
import requests
import json
import ollama
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import config

class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        pass

class OllamaProvider(BaseLLMProvider):
    """Local Ollama LLM Provider."""
    
    def __init__(self, model_name: str = config.OLLAMA_MODEL):
        self.model_name = model_name

    def generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=formatted_messages,
                options={"temperature": 0.2}
            )
            return response["message"]["content"]
        except Exception as e:
            return f"❌ Error from Ollama Provider ({self.model_name}): {str(e)}"

class CloudflareWorkerProvider(BaseLLMProvider):
    """Cloudflare Workers AI REST API LLM Provider.
    Uses efficient models like @cf/meta/llama-3.1-8b-instruct to conserve free tier neurons.
    """
    
    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.account_id = account_id or config.CLOUDFLARE_ACCOUNT_ID
        self.api_token = api_token or config.CLOUDFLARE_API_TOKEN
        self.model_name = model_name or config.CLOUDFLARE_MODEL

    def generate(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        if not self.account_id or not self.api_token:
            return (
                "⚠️ Cloudflare Credentials Missing! Please set `CLOUDFLARE_ACCOUNT_ID` and "
                "`CLOUDFLARE_API_TOKEN` in `.env` or in the settings UI."
            )

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)

        payload = {
            "messages": formatted_messages,
            "temperature": 0.2,
            "max_tokens": 1024
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if data.get("success"):
                    result = data.get("result", {})
                    if "response" in result:
                        return result["response"]
                    elif "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        return json.dumps(result)
                else:
                    errors = data.get("errors", [])
                    return f"❌ Cloudflare Workers AI API Error: {errors}"
            else:
                return f"❌ HTTP Error {res.status_code} from Cloudflare API: {res.text}"
        except Exception as e:
            return f"❌ Cloudflare Workers AI Request Exception: {str(e)}"

class LLMFactory:
    """Factory to instantiate LLM Providers."""
    
    @staticmethod
    def get_provider(
        provider_name: str = config.DEFAULT_PROVIDER,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseLLMProvider:
        name = provider_name.lower()
        if name in ["cloudflare", "cf", "cloudflare_workers"]:
            return CloudflareWorkerProvider(
                account_id=account_id,
                api_token=api_token,
                model_name=model_name or config.CLOUDFLARE_MODEL
            )
        elif name in ["ollama", "local"]:
            return OllamaProvider(
                model_name=model_name or config.OLLAMA_MODEL
            )
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider_name}'. Choose 'cloudflare' or 'ollama'.")
