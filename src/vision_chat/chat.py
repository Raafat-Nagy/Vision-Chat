from collections.abc import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from vision_chat.images import image_to_data_url
from vision_chat.memory import get_session_history
from vision_chat.model import model

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful vision assistant. Answer questions based on the provided image and conversation.",
        ),
        MessagesPlaceholder(variable_name="history"),
        # MessagesPlaceholder, NOT ("human", "{input}"): a string template
        # would stringify multimodal HumanMessages and silently drop the image.
        MessagesPlaceholder(variable_name="input"),
    ]
)


def build_chat_chain(model: BaseChatModel) -> RunnableWithMessageHistory:
    """Wire prompt + model under per-session history."""
    return RunnableWithMessageHistory(
        prompt | model,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


chat = build_chat_chain(model)


def build_user_message(
    question: str,
    image: str | None = None,
) -> HumanMessage:
    """Build the user message: text only, or text + image (first turn)."""
    content: list[dict] = [
        {
            "type": "text",
            "text": question,
        }
    ]

    if image:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image),
                },
            }
        )

    return HumanMessage(content=content)


async def astream_answer(
    session_id: str,
    question: str,
    image: str | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """Yield the answer to one question as (kind, token) pairs.

    kind is "thinking" for reasoning content (when the model/endpoint
    exposes it, e.g. llama.cpp `reasoning_content`) or "answer" for the
    final response. Pass `image` only on the FIRST question of a session:
    it is embedded in that message, stored in the session history, and
    replayed by RunnableWithMessageHistory on every later turn.
    """
    message = build_user_message(question, image)

    async for chunk in chat.astream(
        {"input": [message]},
        config={"configurable": {"session_id": session_id}},
    ):
        reasoning = chunk.additional_kwargs.get("reasoning_content")
        if reasoning:
            yield "thinking", reasoning
        if isinstance(chunk.content, str) and chunk.content:
            yield "answer", chunk.content


async def get_answer(
    session_id: str,
    question: str,
    image: str | None = None,
) -> str:
    """Non-streaming fallback for servers that do not support streaming."""
    message = build_user_message(question, image)

    response = await chat.ainvoke(
        {"input": [message]},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content
