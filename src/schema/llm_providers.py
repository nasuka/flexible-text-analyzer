"""LLMプロバイダーとモデル定義"""

from enum import Enum


class LLMProvider(Enum):
    """LLMプロバイダーの定義"""

    OPENAI = "openai"
    OPENROUTER = "openrouter"

    @classmethod
    def get_provider_names(cls) -> list[str]:
        """プロバイダー名のリストを取得"""
        return [provider.value for provider in cls]

    def get_display_name(self) -> str:
        """プロバイダーの表示名を取得"""
        display_names = {
            self.OPENAI: "OpenAI",
            self.OPENROUTER: "OpenRouter",
        }
        return display_names.get(self, self.value)


class LLMModel(Enum):
    """利用可能なLLMモデルの定義"""

    # OpenAI Models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # OpenRouter Models (Gemini)
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash-preview-05-20"
    GEMINI_2_5_PRO = "google/gemini-2.5-pro-preview"

    @classmethod
    def get_models_by_provider(cls, provider: LLMProvider) -> list["LLMModel"]:
        """プロバイダー別のモデルリストを取得"""
        provider_models = {
            LLMProvider.OPENAI: [
                cls.GPT_4O,
                cls.GPT_4O_MINI,
            ],
            LLMProvider.OPENROUTER: [
                cls.GEMINI_2_5_FLASH,
                cls.GEMINI_2_5_PRO,
            ],
        }
        return provider_models.get(provider, [])

    @classmethod
    def get_model_names_by_provider(cls, provider: LLMProvider) -> list[str]:
        """プロバイダー別のモデル名リストを取得"""
        models = cls.get_models_by_provider(provider)
        return [model.value for model in models]

    def get_provider(self) -> LLMProvider:
        """このモデルのプロバイダーを取得"""
        if self in [self.GPT_4O, self.GPT_4O_MINI]:
            return LLMProvider.OPENAI
        elif self in [self.GEMINI_2_5_FLASH, self.GEMINI_2_5_PRO]:
            return LLMProvider.OPENROUTER
        else:
            raise ValueError(f"Unknown provider for model: {self.value}")

    def get_display_name(self) -> str:
        """モデルの表示名を取得"""
        display_names = {
            self.GPT_4O: "GPT-4o",
            self.GPT_4O_MINI: "GPT-4o Mini",
            self.GEMINI_2_5_FLASH: "Gemini 2.5 Flash",
            self.GEMINI_2_5_PRO: "Gemini 2.5 Pro",
        }
        return display_names.get(self, self.value)

    def supports_structured_output(self) -> bool:
        """Structured Outputサポートの確認"""
        # OpenAIモデルはStructured Outputをサポート
        openai_models = [self.GPT_4O, self.GPT_4O_MINI]

        # Geminiモデルは現在Structured Outputを直接サポートしていないため
        # JSON形式でのレスポンスを要求する形で対応
        gemini_models = [self.GEMINI_2_5_FLASH, self.GEMINI_2_5_PRO]

        return self in openai_models or self in gemini_models

    def get_max_tokens(self) -> int | None:
        """モデルの最大トークン数を取得"""
        max_tokens = {
            self.GPT_4O: 128000,
            self.GPT_4O_MINI: 128000,
            self.GEMINI_2_5_FLASH: 1000000,  # 1M tokens
            self.GEMINI_2_5_PRO: 2000000,  # 2M tokens
        }
        return max_tokens.get(self)


def get_api_base_url(provider: LLMProvider) -> str:
    """プロバイダーのAPI Base URLを取得"""
    urls = {
        LLMProvider.OPENAI: "https://api.openai.com/v1",
        LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    }
    return urls.get(provider, "")


def get_required_headers(provider: LLMProvider, api_key: str) -> dict[str, str]:
    """プロバイダー別の必要なヘッダーを取得"""
    if provider == LLMProvider.OPENAI:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    elif provider == LLMProvider.OPENROUTER:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mirai-analyzer.local",  # Optional: サイト識別用
            "X-Title": "Mirai Analyzer",  # Optional: アプリ名
        }
    else:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


def validate_api_key_format(provider: LLMProvider, api_key: str) -> bool:
    """プロバイダー別のAPIキー形式検証"""
    if not api_key:
        return False

    if provider == LLMProvider.OPENAI:
        return api_key.startswith("sk-")
    elif provider == LLMProvider.OPENROUTER:
        return api_key.startswith("sk-or-")
    else:
        return len(api_key) > 10  # 基本的な長さチェック
