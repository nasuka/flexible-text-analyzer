"""共通LLMクライアント"""

from typing import Any

import openai
import streamlit as st
from pydantic import BaseModel


class LLMClient:
    """OpenAI APIの共通クライアント"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        LLMクライアントの初期化

        Args:
            api_key: OpenAI APIキー
            model: 使用するモデル名
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def structured_completion(
        self,
        prompt: str,
        response_format: type[BaseModel],
        system_message: str = "",
        temperature: float = 0.3,
    ) -> BaseModel | None:
        """
        Structured Outputを使用してLLMから構造化データを取得

        Args:
            prompt: ユーザープロンプト
            response_format: レスポンスのPydanticモデル
            system_message: システムメッセージ
            temperature: 生成の確率的変動性

        Returns:
            パースされたPydanticモデルまたはNone
        """
        try:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": prompt})

            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
            )

            return response.choices[0].message.parsed

        except Exception as e:
            st.error(f"LLMリクエストでエラーが発生しました: {str(e)}")
            return None

    def batch_structured_completion(
        self,
        prompts: list[str],
        response_format: type[BaseModel],
        system_message: str = "",
        temperature: float = 0.3,
    ) -> list[BaseModel | None]:
        """
        複数のプロンプトに対してバッチでStructured Outputを取得

        Args:
            prompts: プロンプトのリスト
            response_format: レスポンスのPydanticモデル
            system_message: システムメッセージ
            temperature: 生成の確率的変動性

        Returns:
            パースされたPydanticモデルのリスト
        """
        results = []

        for prompt in prompts:
            result = self.structured_completion(
                prompt=prompt,
                response_format=response_format,
                system_message=system_message,
                temperature=temperature,
            )
            results.append(result)

        return results

    def simple_completion(
        self,
        prompt: str,
        system_message: str = "",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str | None:
        """
        通常のテキスト生成（非構造化）

        Args:
            prompt: ユーザープロンプト
            system_message: システムメッセージ
            temperature: 生成の確率的変動性
            max_tokens: 最大トークン数

        Returns:
            生成されたテキストまたはNone
        """
        try:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            st.error(f"LLMリクエストでエラーが発生しました: {str(e)}")
            return None

    def get_model_info(self) -> dict[str, Any]:
        """
        使用中のモデル情報を取得

        Returns:
            モデル情報の辞書
        """
        return {
            "model": self.model,
            "client_type": "OpenAI",
            "api_base": self.client.base_url,
        }


class LLMError(Exception):
    """LLM関連のカスタム例外"""

    pass


def create_llm_client(api_key: str, model: str = "gpt-4o") -> LLMClient:
    """
    LLMクライアントのファクトリー関数

    Args:
        api_key: OpenAI APIキー
        model: 使用するモデル名

    Returns:
        LLMClientインスタンス

    Raises:
        LLMError: APIキーが無効な場合
    """
    if not api_key:
        raise LLMError("APIキーが指定されていません")

    return LLMClient(api_key=api_key, model=model)


def validate_api_key(api_key: str) -> bool:
    """
    APIキーの有効性を検証

    Args:
        api_key: 検証するAPIキー

    Returns:
        APIキーが有効かどうか
    """
    if not api_key or not api_key.startswith("sk-"):
        return False

    try:
        client = openai.OpenAI(api_key=api_key)
        # 簡単なテストリクエストを送信
        client.models.list()
        return True
    except Exception:
        return False
