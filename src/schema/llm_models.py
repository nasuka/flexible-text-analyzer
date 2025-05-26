from enum import Enum


class LLMModels(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O4_MINI = "o4-mini"

    @classmethod
    def get_model_names(cls) -> list[str]:
        return [model.value for model in cls]