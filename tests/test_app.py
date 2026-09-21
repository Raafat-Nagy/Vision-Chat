"""Regression tests for the Chainlit lifecycle in app.py (no server)."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


class FakeAskFileMessage:
    def __init__(self, response, **kwargs):
        self.response = response

    async def send(self):
        return self.response


class FakeMessage:
    sent: list["FakeMessage"] = []  # noqa: RUF012 - test fake

    def __init__(self, content="", elements=None):
        self.content = content
        self.elements = elements

    async def send(self):
        FakeMessage.sent.append(self)


def run_on_chat_start(response) -> tuple[dict, list[FakeMessage]]:
    session: dict = {}
    FakeMessage.sent = []

    with (
        patch.object(app.cl, "AskFileMessage", lambda **kw: FakeAskFileMessage(response, **kw)),
        patch.object(
            app.cl,
            "user_session",
            SimpleNamespace(set=lambda k, v: session.update({k: v}), get=session.get),
        ),
        patch.object(app.cl, "Message", FakeMessage),
        patch.object(app.cl, "Image", lambda **kw: SimpleNamespace(**kw)),
    ):
        asyncio.run(app.on_chat_start())

    return session, FakeMessage.sent


def test_upload_sets_new_session_and_image() -> None:
    upload = [SimpleNamespace(name="car.jpg", path="/tmp/upload/car.jpg")]

    session, sent = run_on_chat_start(upload)

    assert session["image_path"] == "/tmp/upload/car.jpg"
    assert session["session_id"]
    assert "Image received" in sent[-1].content


def test_no_upload_does_not_start_a_session() -> None:
    session, sent = run_on_chat_start(None)

    assert "session_id" not in session
    assert "No image" in sent[-1].content


# --- streaming / thinking / fallback routing ---


class FakeTokenMessage:
    def __init__(self, content="", **kwargs):
        self.content = content
        self.tokens: list[str] = []

    async def stream_token(self, token: str):
        self.tokens.append(token)
        self.content += token

    async def send(self):
        FakeMessage.sent.append(self)

    async def update(self):
        pass


class FakeStep(FakeTokenMessage):
    pass


def run_on_message(behavior, question="Think about this") -> dict:
    """Run on_message with astream_answer patched to `behavior`.

    behavior is either an async iterable of (kind, token) pairs, or the
    string "fail" to make streaming raise (fallback path).
    """
    session = {
        "session_id": "ui-test",
        "image_path": "/tmp/upload/car.jpg",
    }
    steps: list[FakeStep] = []
    messages_sent: list = []

    async def fake_astream(*args, **kwargs):
        if behavior == "fail":
            raise RuntimeError("no streaming support")
        for kind, token in behavior:
            yield kind, token

    async def fake_get_answer(*args, **kwargs):
        return "fallback answer"

    user_message = SimpleNamespace(content=question)

    with (
        patch.object(app, "astream_answer", fake_astream),
        patch.object(app, "get_answer", fake_get_answer),
        patch.object(
            app.cl,
            "user_session",
            SimpleNamespace(set=lambda k, v: session.update({k: v}), get=session.get),
        ),
        patch.object(app, "get_session_history", lambda sid: SimpleNamespace(messages=[AIMessage(content="hi")])),
        patch.object(app.cl, "Message", FakeTokenMessage),
        patch.object(app.cl, "Step", FakeStep),
    ):
        # capture steps created via cl.Step(...)
        original_init = FakeStep.__init__

        def step_init(self, *a, **kw):
            original_init(self, *a, **kw)
            steps.append(self)

        FakeStep.__init__ = step_init
        FakeMessage.sent = []
        original_send = FakeTokenMessage.send

        async def track_send(self):
            if isinstance(self, FakeStep) and self in steps:
                return  # steps tracked separately
            messages_sent.append(self)

        FakeTokenMessage.send = track_send
        try:
            asyncio.run(app.on_message(user_message))
        finally:
            FakeTokenMessage.send = original_send
            FakeStep.__init__ = original_init

    return {"steps": steps, "messages": messages_sent, "session": session}


def test_streaming_routes_thinking_to_step_and_answer_to_message() -> None:
    tokens = [
        ("thinking", "Let me look... "),
        ("thinking", "a red car."),
        ("answer", "I see "),
        ("answer", "a red car."),
    ]

    result = run_on_message(tokens)

    assert len(result["steps"]) == 1, "one Thinking step"
    step = result["steps"][0]
    assert "".join(step.tokens) == "Let me look... a red car."
    answer_messages = [m for m in result["messages"] if isinstance(m, FakeTokenMessage)]
    assert "".join(answer_messages[0].tokens) == "I see a red car."


def test_streaming_failure_falls_back_to_non_streaming() -> None:
    result = run_on_message("fail")

    assert result["steps"] == [], "no Thinking step on fallback"
    contents = [m.content for m in result["messages"]]
    assert "fallback answer" in contents


def test_streaming_without_thinking_creates_no_step() -> None:
    result = run_on_message([("answer", "just "), ("answer", "an answer")])

    assert result["steps"] == []
    answer_messages = [m for m in result["messages"] if isinstance(m, FakeTokenMessage)]
    assert "".join(answer_messages[0].tokens) == "just an answer"
