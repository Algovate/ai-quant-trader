"""
简化的LLM模型管理器 - 只支持OpenRouter
"""

import os
import requests
from typing import Dict, List, Optional, Any
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class OpenRouterProvider:
    """OpenRouter API提供商"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "AI Trading System"
        }

    def call_model(self, prompt: str, model: str, **kwargs) -> str:
        """调用OpenRouter API"""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": kwargs.get("system_prompt", "You are a financial analyst.")},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.3)
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=kwargs.get("timeout", 30)
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                raise Exception(f"API call failed: {response.status_code}")

        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """获取OpenRouter可用模型"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 200:
                models_data = response.json()
                return [model["id"] for model in models_data.get("data", [])]
            else:
                logger.warning(f"Failed to fetch models: {response.status_code}")
                return self._get_default_models()

        except Exception as e:
            logger.warning(f"Error fetching models: {e}")
            return self._get_default_models()

    def _get_default_models(self) -> List[str]:
        """获取默认模型列表"""
        return [
            "deepseek/deepseek-chat-v3-0324",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet"
        ]


class ModelManager:
    """简化的模型管理器 - 只支持OpenRouter"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY not found in environment variables")

        self.provider = OpenRouterProvider(self.api_key)
        logger.info("Model manager initialized with OpenRouter provider")

    def call_model(self, prompt: str, model: str, **kwargs) -> str:
        """调用模型"""
        return self.provider.call_model(prompt, model, **kwargs)

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self.provider.get_available_models()

    def get_recommended_models(self) -> Dict[str, str]:
        """获取推荐模型配置（简化版）"""
        return {
            "deepseek-chat-v3": "DeepSeek最新模型，推理能力强",
            "gpt-4o-mini": "OpenAI快速模型，成本效益高",
            "claude-3.5-sonnet": "Anthropic最强推理模型"
        }

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        default_config = {
            "temperature": 0.3,
            "max_tokens": 500,
            "system_prompt": "You are a professional financial analyst."
        }

        model_configs = {
            "deepseek/deepseek-chat-v3-0324": {
                "temperature": 0.3,
                "max_tokens": 1000,
                "system_prompt": "You are an expert quantitative analyst specializing in stock market analysis."
            },
            "openai/gpt-4o-mini": {
                "temperature": 0.3,
                "max_tokens": 300,
                "system_prompt": "Provide quick, concise market analysis."
            },
            "anthropic/claude-3.5-sonnet": {
                "temperature": 0.1,
                "max_tokens": 1000,
                "system_prompt": "You are a professional financial analyst with expertise in quantitative trading."
            }
        }

        return model_configs.get(model_name, default_config)

    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            models = self.get_available_models()
            logger.info(f"Successfully connected to OpenRouter. Available models: {len(models)}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OpenRouter: {e}")
            return False


def create_model_manager(api_key: str = None) -> ModelManager:
    """创建模型管理器实例"""
    return ModelManager(api_key)


if __name__ == "__main__":
    try:
        manager = create_model_manager()

        print("OpenRouter Model Manager")
        print("="*40)

        if manager.test_connection():
            print("✓ Successfully connected to OpenRouter")

            print("\nRecommended models:")
            recommended = manager.get_recommended_models()
            for name, description in recommended.items():
                print(f"  {name}: {description}")

            print("\nAvailable models (first 10):")
            models = manager.get_available_models()
            for model in models[:10]:
                print(f"  - {model}")
        else:
            print("✗ Failed to connect to OpenRouter")

    except Exception as e:
        print(f"Error: {e}")