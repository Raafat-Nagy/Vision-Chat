"""Chainlit web UI for Vision Chat.

Every LangChain detail lives in `vision_chat.chat`; this module only
handles the UI lifecycle:

- on_chat_start: the user uploads ONE image -> a NEW session id
  (Chainlit's "+ New Chat" triggers on_chat_start again).
- on_message: streams the answer chunk-by-chunk into a message; reasoning
  content, when the model exposes it, streams into a separate collapsible
  "Thinking" step. Falls back to a single non-streaming reply if the
  server does not support streaming.
"""

import logging
import uuid

import chainlit as cl

from vision_chat.chat import astream_answer, get_answer
from vision_chat.memory import get_session_history

logger = logging.getLogger("vision_chat.ui")


@cl.on_chat_start
async def on_chat_start() -> None:
    uploaded = await cl.AskFileMessage(
        content="Upload an image to chat about it.",
        accept=["image/png", "image/jpeg", "image/webp"],
        max_size_mb=20,
        max_files=1,
    ).send()

    if not uploaded:
        await cl.Message(
            content="No image was provided. Reload the page to start again."
        ).send()
        return

    image = uploaded[0]  # Chainlit 2.x returns a list of AskFileResponse

    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("image_path", image.path)

    await cl.Message(
        content=(
            "Image received! Ask me anything about it.\n\n"
            "Use **+ New Chat** to start a conversation about a different image."
        ),
        elements=[cl.Image(name=image.name, path=image.path, display="inline")],
    ).send()


async def stream_response(
    session_id: str,
    question: str,
    image: str | None,
) -> str:
    """Stream reasoning (if any) into a Thinking step and the answer into
    a message. Returns the final answer text."""
    answer_message = cl.Message(content="")
    thinking_step: cl.Step | None = None

    async for kind, token in astream_answer(session_id, question, image):
        if kind == "thinking":
            if thinking_step is None:
                thinking_step = cl.Step(name="Thinking", type="llm")
                await thinking_step.send()
            await thinking_step.stream_token(token)
        else:
            await answer_message.stream_token(token)

    if thinking_step is not None:
        await thinking_step.update()
    await answer_message.send()

    return answer_message.content


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session_id = cl.user_session.get("session_id")
    image_path = cl.user_session.get("image_path")

    if not session_id:
        await cl.Message(
            content="Start a new chat and upload an image first."
        ).send()
        return

    # The image is attached only on the first question of the session -
    # after that it already lives in the session history.
    first_turn = not get_session_history(session_id).messages
    image = image_path if first_turn else None

    try:
        await stream_response(session_id, message.content, image)
    except Exception:
        # Streaming failed (e.g. the server does not support it) - fall back
        # to one non-streaming request. The failed stream adds nothing to the
        # session history, so this retry starts from a clean state.
        logger.exception("Streaming failed; falling back to non-streaming request")
        answer = await get_answer(session_id, message.content, image)
        await cl.Message(content=answer).send()
