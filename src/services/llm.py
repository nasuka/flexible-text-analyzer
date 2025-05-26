"""共通LLMクライアント"""

import json
from typing import Any

import openai
import streamlit as st
from pydantic import BaseModel

from schema.llm_providers import (
    LLMModel,
    LLMProvider,
    get_api_base_url,
    get_required_headers,
    validate_api_key_format,
)


class LLMClient:
    """マルチプロバイダー対応のLLMクライアント"""

    def __init__(
        self,
        api_key: str,
        model: str | LLMModel = "gpt-4o",
        provider: LLMProvider | None = None,
    ):
        """
        LLMクライアントの初期化

        Args:
            api_key: APIキー
            model: 使用するモデル名またはLLMModelインスタンス
            provider: LLMプロバイダー（指定しない場合はモデルから自動判定）
        """
        # モデル情報の正規化
        if isinstance(model, str):
            try:
                self.model_enum = LLMModel(model)
            except ValueError:
                # 後方互換性のため、既存の文字列モデル名もサポート
                self.model_enum = (
                    LLMModel.GPT_4O if model == "gpt-4o" else LLMModel.GPT_4O_MINI
                )
        else:
            self.model_enum = model

        self.model = self.model_enum.value

        # プロバイダーの決定
        if provider is None:
            self.provider = self.model_enum.get_provider()
        else:
            self.provider = provider

        self.api_key = api_key

        # プロバイダー別のクライアント初期化
        if self.provider == LLMProvider.OPENAI:
            self.client = openai.OpenAI(api_key=api_key)
        elif self.provider == LLMProvider.OPENROUTER:
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=get_api_base_url(self.provider),
                default_headers=get_required_headers(self.provider, api_key),
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # APIキー形式の検証
        if not validate_api_key_format(self.provider, api_key):
            st.warning(
                f"⚠️ {self.provider.get_display_name()}のAPIキー形式が正しくない可能性があります"
            )

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

            # プロバイダー別の処理
            if self.provider == LLMProvider.OPENAI:
                # OpenAIはネイティブなStructured Outputをサポート
                messages.append({"role": "user", "content": prompt})

                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                )

                return response.choices[0].message.parsed

            else:
                # OpenRouter（Gemini等）はJSON形式でのレスポンスを要求
                schema = response_format.model_json_schema()
                json_prompt = f"""
{prompt}

重要: 以下のJSON形式で正確に回答してください。余計な説明は不要です。

JSON Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

レスポンス:
"""
                messages.append({"role": "user", "content": json_prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )

                # JSONレスポンスをパース
                response_text = response.choices[0].message.content

                # JSONの抽出（```json...```で囲まれている場合の対応）
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    start = response_text.find("```") + 3
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()

                try:
                    json_data = json.loads(response_text)
                    return response_format(**json_data)
                except (json.JSONDecodeError, TypeError) as e:
                    st.error(f"JSONパースエラー: {str(e)}\nレスポンス: {response_text}")
                    return None

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
            "model_display_name": self.model_enum.get_display_name(),
            "provider": self.provider.value,
            "provider_display_name": self.provider.get_display_name(),
            "api_base": self.client.base_url,
            "supports_structured_output": self.model_enum.supports_structured_output(),
            "max_tokens": self.model_enum.get_max_tokens(),
        }


class LLMError(Exception):
    """LLM関連のカスタム例外"""

    pass


def create_llm_client(
    api_key: str, model: str | LLMModel = "gpt-4o", provider: LLMProvider | None = None
) -> LLMClient:
    """
    LLMクライアントのファクトリー関数

    Args:
        api_key: APIキー
        model: 使用するモデル名またはLLMModelインスタンス
        provider: LLMプロバイダー（指定しない場合はモデルから自動判定）

    Returns:
        LLMClientインスタンス

    Raises:
        LLMError: APIキーが無効な場合
    """
    if not api_key:
        raise LLMError("APIキーが指定されていません")

    return LLMClient(api_key=api_key, model=model, provider=provider)


def validate_api_key(api_key: str, provider: LLMProvider = LLMProvider.OPENAI) -> bool:
    """
    APIキーの有効性を検証

    Args:
        api_key: 検証するAPIキー
        provider: LLMプロバイダー

    Returns:
        APIキーが有効かどうか
    """
    if not validate_api_key_format(provider, api_key):
        return False

    try:
        if provider == LLMProvider.OPENAI:
            client = openai.OpenAI(api_key=api_key)
        elif provider == LLMProvider.OPENROUTER:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=get_api_base_url(provider),
                default_headers=get_required_headers(provider, api_key),
            )
        else:
            return False

        # 簡単なテストリクエストを送信
        client.models.list()
        return True
    except Exception:
        return False
