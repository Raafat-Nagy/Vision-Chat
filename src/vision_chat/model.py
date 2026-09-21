from langchain_openai import ChatOpenAI

from vision_chat.config import settings


class LlamaChatOpenAI(ChatOpenAI):
    """ChatOpenAI that preserves llama.cpp's `reasoning_content`.

    langchain-openai does not extract `reasoning_content` from streaming
    deltas (documented behavior). llama.cpp servers running a thinking
    model with `--reasoning-format auto|deepseek` stream reasoning there,
    so keep it in `additional_kwargs` for the UI to display separately.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None:
            return None

        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get(
            "choices", []
        )
        if choices:
            reasoning = choices[0].get("delta", {}).get("reasoning_content")
            if reasoning:
                generation.message.additional_kwargs["reasoning_content"] = reasoning

        return generation


model = LlamaChatOpenAI(
    base_url=settings.LLAMA_BASE_URL,
    api_key=settings.LLAMA_API_KEY,
    model=settings.MODEL_NAME,
    max_tokens=settings.MAX_TOKENS,
    temperature=0,
)
