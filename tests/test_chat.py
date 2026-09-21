"""Tests for the core chat logic. No server, no real model."""

import uuid

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from vision_chat.chat import build_chat_chain, build_user_message
from vision_chat.memory import get_session_history

SENT: list[list] = []


class FakeModel(BaseChatModel):
    """Records the messages it receives instead of calling a server."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        SENT.append(list(messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )

    @property
    def _llm_type(self) -> str:
        return "fake"


chat = build_chat_chain(FakeModel())


def unique_session_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


def image_message(question: str = "What do you see?") -> HumanMessage:
    return build_user_message(question, image="assets/red_car.jpg")


def count_images(messages: list) -> int:
    total = 0
    for m in messages:
        if isinstance(m.content, list) and any(
            p.get("type") == "image_url" for p in m.content
        ):
            total += 1
    return total


def invoke(session_id: str, message: HumanMessage) -> None:
    chat.invoke(
        {"input": [message]},
        config={"configurable": {"session_id": session_id}},
    )


def test_history_accessor_creates_history_automatically() -> None:
    session_id = unique_session_id()

    history = get_session_history(session_id)

    assert history.messages == []
    assert get_session_history(session_id) is history


def test_first_turn_sends_system_prompt_and_image() -> None:
    session_id = unique_session_id()

    invoke(session_id, image_message())

    received = SENT[-1]
    assert received[0] == SystemMessage(
        content="You are a helpful vision assistant. "
        "Answer questions based on the provided image and conversation."
    )
    assert count_images(received) == 1
    assert len(received) == 2


def test_messages_are_preserved_within_a_session() -> None:
    session_id = unique_session_id()

    invoke(session_id, image_message())
    invoke(session_id, HumanMessage(content="What color is it?"))
    invoke(session_id, HumanMessage(content="Any damage?"))

    received = SENT[-1]
    # system + (human+image, ai) + (human, ai) + question
    assert len(received) == 6
    assert count_images(received) == 1, "image appears once, from history"
    assert count_images([received[1]]) == 1, "image sits in the first user turn"

    assert len(get_session_history(session_id).messages) == 6
    assert isinstance(get_session_history(session_id).messages[-1], AIMessage)


def test_different_sessions_are_isolated() -> None:
    session_a = unique_session_id()
    session_b = unique_session_id()

    invoke(session_a, image_message())
    invoke(session_b, HumanMessage(content="hello"))

    assert len(SENT[-1]) == 2, "session B must not receive session A's history"
    assert count_images(SENT[-1]) == 0
    assert len(get_session_history(session_a).messages) == 2
    assert len(get_session_history(session_b).messages) == 2


def test_new_image_starts_a_new_clean_session() -> None:
    session_a = unique_session_id()
    session_b = unique_session_id()

    invoke(session_a, image_message("What do you see?"))
    invoke(session_b, image_message("What is in this picture?"))

    received = SENT[-1]
    assert count_images(received) == 1, "only the new image is present"
    assert count_images(SENT[-2]) == 1
    assert len(get_session_history(session_b).messages) == 2


def test_chat_logic_runs_without_a_real_server() -> None:
    session_id = unique_session_id()

    invoke(session_id, image_message())
    invoke(session_id, HumanMessage(content="What color is it?"))

    assert SENT[-1][-1].content == "What color is it?"
    assert get_session_history(session_id).messages[-1].content == "ok"


def test_image_message_format() -> None:
    message = image_message()

    assert isinstance(message, HumanMessage)
    assert message.content[0] == {"type": "text", "text": "What do you see?"}
    image_part = message.content[1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_text_only_message_format() -> None:
    message = build_user_message("What color is it?")

    assert message.content == [{"type": "text", "text": "What color is it?"}]


def test_missing_image_file_raises(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        build_user_message("What do you see?", image=str(tmp_path / "nope.jpg"))


# --- streaming ---

import asyncio

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk


class FakeStreamModel(BaseChatModel):
    """Yields reasoning chunks then answer chunks, like llama.cpp."""

    fail: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.fail:
            raise RuntimeError("server died")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        last_user = next(
            (m for m in reversed(messages) if m.__class__ is HumanMessage), None
        )
        thinks = bool(last_user and "think" in str(last_user.content).lower())
        pieces = [
            *(
                [
                    {"reasoning_content": "The image shows "},
                    {"reasoning_content": "a red car."},
                ]
                if thinks
                else []
            ),
            {"content": "I see "},
            {"content": "a red car."},
        ]
        for piece in pieces:
            if self.fail and piece.get("content"):
                raise RuntimeError("server died mid-answer")
            chunk = AIMessageChunk(
                content=piece.get("content", ""),
                additional_kwargs=(
                    {"reasoning_content": piece["reasoning_content"]}
                    if "reasoning_content" in piece
                    else {}
                ),
            )
            if run_manager is not None:  # None on the first probe chunk
                run_manager.on_llm_new_token(chunk.content, chunk=chunk)
            yield ChatGenerationChunk(message=chunk)

    @property
    def _llm_type(self) -> str:
        return "fake-stream"


def collect(session_id: str, question: str, image=None):
    from vision_chat.chat import astream_answer

    async def run():
        return [
            (kind, token)
            async for kind, token in astream_answer(session_id, question, image)
        ]

    return asyncio.run(run())


def test_streaming_separates_thinking_from_answer() -> None:
    stream_chat = build_chat_chain(FakeStreamModel())
    SENT.clear()
    globals()["_old_chat"] = None

    import vision_chat.chat as chat_module

    saved = chat_module.chat
    chat_module.chat = stream_chat
    try:
        session_id = unique_session_id()
        pieces = collect(
            session_id, "Think and tell me what you see?", image="assets/red_car.jpg"
        )
    finally:
        chat_module.chat = saved

    thinking = "".join(t for k, t in pieces if k == "thinking")
    answer = "".join(t for k, t in pieces if k == "answer")
    assert thinking == "The image shows a red car."
    assert answer == "I see a red car."
    assert pieces[0][0] == "thinking"

    # history still records the full turn once
    history = get_session_history(session_id).messages
    assert len(history) == 2
    assert history[-1].content == "I see a red car."


def test_streaming_without_reasoning_only_answers() -> None:
    import vision_chat.chat as chat_module

    saved = chat_module.chat
    chat_module.chat = build_chat_chain(FakeStreamModel())
    try:
        pieces = collect(unique_session_id(), "plain question")
    finally:
        chat_module.chat = saved

    assert all(k == "answer" for k, _ in pieces)


def test_failed_stream_does_not_corrupt_history() -> None:
    import vision_chat.chat as chat_module

    failing = FakeStreamModel()
    failing.fail = True
    saved = chat_module.chat
    chat_module.chat = build_chat_chain(failing)
    try:
        session_id = unique_session_id()
        invoke(session_id, HumanMessage(content="first ok"))
        before = len(get_session_history(session_id).messages)

        with pytest.raises(RuntimeError):
            collect(session_id, "this will fail")

        after = get_session_history(session_id).messages
        assert len(after) == before == 2, "failed stream must not touch history"
        assert after[-1].content == "ok"
    finally:
        chat_module.chat = saved


def test_non_streaming_fallback_answers_and_records() -> None:
    import asyncio

    import vision_chat.chat as chat_module

    saved = chat_module.chat
    chat_module.chat = build_chat_chain(FakeStreamModel())
    try:
        from vision_chat.chat import get_answer

        session_id = unique_session_id()
        answer = asyncio.run(
            get_answer(session_id, "What do you see?", image="assets/red_car.jpg")
        )
        assert answer == "ok"
        history = get_session_history(session_id).messages
        assert len(history) == 2
        assert history[-1].content == "ok"
    finally:
        chat_module.chat = saved
