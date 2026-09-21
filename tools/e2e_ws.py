"""End-to-end check: real Chainlit server + real socket.io client + mock model.

Mirrors the browser protocol: connect -> connection_successful -> 'ask'
callback (HTTP upload) -> client_message -> observe streaming events.
Kept under tools/ so it survives workspace resets; run while `make ui`
and `make mock-server` (or a real llama serve) are up:

    uv run python tools/e2e_ws.py
"""

import asyncio
import datetime
import uuid
from pathlib import Path

import httpx
import socketio

BASE = "http://127.0.0.1:8000"
IMAGE = "assets/red_car.jpg"
events: list[tuple[str, dict]] = []
upload_results: list = []

sio = socketio.AsyncClient(logger=False)
sio_session: dict = {}


@sio.on("*")
async def catch_all(event, *args):
    data = args[0] if args else {}
    events.append((event, data))


@sio.on("ask")
async def handle_ask(payload):
    """Server asks for a file (AskFileMessage). Upload over HTTP, return [FileDict]."""
    spec = payload["spec"]  # payload = {"msg": step_dict, "spec": AskFileSpec}
    parent_id = payload["msg"]["parentId"]  # files_spec is keyed by parent id
    print("  [ask] file spec:", spec, "| parent:", parent_id)
    assert spec["type"] == "file"
    data = await asyncio.to_thread(Path(IMAGE).read_bytes)
    async with httpx.AsyncClient() as http:
        r = await http.post(
            f"{BASE}/project/file",
            params={
                "session_id": sio_session["id"],
                "ask_parent_id": parent_id,
            },
            files={"file": ("red_car.jpg", data, "image/jpeg")},
        )
    print("  [ask] upload status:", r.status_code)
    assert r.status_code == 200, r.text
    upload_results.append(r.json())
    return [r.json()]


async def wait_for(predicate, timeout=30, desc=""):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"timeout waiting for {desc}")


def _collect(name_match=None) -> str:
    """Reconstruct a streamed text (first token rides in stream_start.output)."""
    starts = [
        a
        for e, a in events
        if e == "stream_start" and (name_match is None or a.get("name") == name_match)
    ]
    ids = {a["id"] for a in starts}
    head = "".join(a.get("output", "") for a in starts)
    tail = "".join(
        a.get("token", "")
        for e, a in events
        if e == "stream_token" and a.get("id") in ids
    )
    return head + tail


def streamed_answer() -> str:
    """All streamed tokens except Thinking steps."""
    think_ids = {
        a["id"] for e, a in events if e == "stream_start" and a.get("name") == "Thinking"
    }
    return "".join(
        a.get("token", "")
        for e, a in events
        if e == "stream_token" and a.get("id") not in think_ids
    )


def thinking_text() -> str:
    return _collect("Thinking")


def thinking_seen() -> bool:
    return any(a.get("name") == "Thinking" for e, a in events if e == "stream_start")


async def ask_question(text: str):
    events.clear()
    step = {
        "id": str(uuid.uuid4()),
        "createdAt": datetime.datetime.now(datetime.UTC).isoformat(),
        "output": text,
        "type": "user_message",
        "name": "user",
    }
    await sio.emit("client_message", {"message": step, "fileReferences": None})
    # The user-step echo arrives as `new_message` long before the answer
    # finishes streaming, so waiting for it alone races the stream. Wait
    # for the stream to start, then for events to go quiet: no new events
    # for ~250ms (longer than any per-chunk --delay-ms) means the turn is
    # done and every token has been captured.
    await wait_for(
        lambda: any(e == "stream_start" for e, _ in events),
        desc=f"stream start for {text!r}",
    )
    quiet_polls = 0
    while quiet_polls < 5:
        count = len(events)
        await asyncio.sleep(0.05)
        quiet_polls = quiet_polls + 1 if len(events) == count else 0


async def main():
    session_id = str(uuid.uuid4())
    sio_session["id"] = session_id

    await sio.connect(
        BASE,
        auth={
            "sessionId": session_id,
            "threadId": None,
            "userEnv": "{}",
            "clientType": "webapp",
            "chatProfile": None,
        },
        socketio_path="ws/socket.io",
        transports=["websocket"],
    )
    await sio.emit("connection_successful")
    print("== connected")

    await wait_for(lambda: upload_results, desc="upload round-trip")
    print("== image uploaded via AskFileMessage OK")

    # unsolicited upload must be rejected (spontaneous upload disabled)
    data = await asyncio.to_thread(Path(IMAGE).read_bytes)
    async with httpx.AsyncClient() as http:
        r = await http.post(
            f"{BASE}/project/file",
            params={"session_id": session_id},
            files={"file": ("red_car.jpg", data, "image/jpeg")},
        )
    print("== unsolicited upload rejected with:", r.status_code)
    assert r.status_code == 400, r.text

    # Q1: first turn, carries the image
    await ask_question("What do you see?")
    a1 = streamed_answer()
    token_events = [a for e, a in events if e == "stream_token"]
    assert len(token_events) > 5, "Q1 must stream progressively"
    assert "red car" in a1.lower(), a1
    assert "image" in a1.lower(), "mock must confirm the image was received"
    print(f"== Q1 streamed {len(token_events)} tokens -> {a1[:70]!r}")

    # Q2: follow-up, no image
    await ask_question("What color is it?")
    a2 = streamed_answer()
    assert "red" in a2.lower(), a2
    assert not thinking_seen()
    print(f"== Q2 streamed -> {a2[:70]!r}")

    # Q3: thinking question -> separate Thinking step streams reasoning
    await ask_question("Think and tell me what you see")
    a3 = streamed_answer()
    t3 = thinking_text()
    assert thinking_seen(), "Thinking step must stream"
    assert t3.startswith("The user asks"), t3
    assert "red car" in a3.lower(), a3
    print(f"== Q3 thinking -> {t3[:60]!r}")
    print(f"== Q3 answer   -> {a3[:60]!r}")

    await sio.disconnect()
    print("\nE2E WEBSOCKET VERIFICATION PASSED")


if __name__ == "__main__":
    asyncio.run(main())
