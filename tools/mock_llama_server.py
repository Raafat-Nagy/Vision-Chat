"""
Dev-only mock of `llama-server`'s OpenAI-compatible interface.

Lets you test the whole LangChain wiring (model -> prompt -> memory -> UI)
without downloading a vision model. It is STATELESS on purpose: just like
the real llama.cpp server, it receives the full conversation on every
request — session memory lives client-side, in LangChain's history.

Streaming works like llama.cpp with stream=true (SSE chunks, [DONE] terminator).
Questions containing "think"/"why"/"explain" also produce `reasoning_content`
deltas first, like llama.cpp serving a thinking model with
`--reasoning-format auto|deepseek`. Use --delay-ms to make generation slow
enough to watch progressive streaming.

Run:  make mock-server                                   (listens on :8081)
Then: make ui                                            (as usual)
"""

import base64
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_PORT = 8081
DELAY_MS = 0


def describe_payload(messages: list[dict]) -> str:
    """Summarize what the server received, so every reply doubles as a check."""
    image_msgs = [
        i
        for i, m in enumerate(messages)
        if isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"])
    ]

    detail = f"[received {len(messages)} messages; image in message index(es) {image_msgs}]"

    if image_msgs and image_msgs != [1]:
        detail += " (NOTE: real vision models expect the image in the first user message)"

    # prove the data URL is a real, decodable image
    if image_msgs:
        part = next(
            p
            for p in messages[image_msgs[0]]["content"]
            if p.get("type") == "image_url"
        )
        url = part["image_url"]["url"]
        _, b64 = url.split(",", 1)
        header = base64.b64decode(b64[:64] + "==")[:4]
        ok = header.startswith((b"\x89PNG", b"\xff\xd8"))
        detail += f" [image decodes OK: {ok}]"

    return detail


def canned_answer(last_user_text: str) -> str:
    text = last_user_text.lower()

    if "color" in text or "colour" in text:
        return "The car is red."
    if "damage" in text or "dent" in text:
        return "There is a dent on the front door."
    if "what" in text or "see" in text:
        return "I see a red car parked on a residential street."

    return "I am a mock vision model; ask me 'What do you see?', 'What color is it?' or 'Does it have any visible damage?'"


def canned_reasoning(last_user_text: str) -> str:
    """Reasoning text, like a thinking model produces before the answer."""
    return (
        f"The user asks: {last_user_text.strip() or '?'} "
        "Let me examine the image carefully. "
        "It shows a red sedan parked on a street, "
        "and the front door has a visible dent. "
        "I can answer from what I see."
    )


def wants_reasoning(last_user_text: str) -> bool:
    text = last_user_text.lower()
    return "think" in text or "why" in text or "explain" in text


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": "mock-vision-model", "object": "model"}],
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def _send_stream(self, model: str, answer: str, reasoning: str | None) -> None:
        """Stream as SSE chunks, like llama-server with stream=true.

        Reasoning deltas (if any) are sent first as `reasoning_content`,
        then the answer as `content` - the llama.cpp thinking-model shape.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if reasoning:
            for piece in reasoning.split(" "):
                self._send_delta(model, {"reasoning_content": piece + " "})

        words = answer.split(" ")

        for i, word in enumerate(words):
            piece = word + (" " if i < len(words) - 1 else "")
            delta: dict = {"content": piece}
            if i == 0:
                delta["role"] = "assistant"
            self._send_delta(model, delta)

        self._send_event(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")

    def _send_delta(self, model: str, delta: dict) -> None:
        if DELAY_MS:
            time.sleep(DELAY_MS / 1000)
        self._send_event(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )

    def _send_event(self, payload: dict) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length))

        if not self.path.startswith("/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return

        messages = data.get("messages", [])
        stream = bool(data.get("stream"))
        print(f"[mock] POST /v1/chat/completions messages={len(messages)} stream={stream}")
        last_user = next(
            (
                m["content"]
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        if isinstance(last_user, list):
            last_user = next(
                (p.get("text", "") for p in last_user if p.get("type") == "text"),
                "",
            )

        answer = f"{canned_answer(last_user)} {describe_payload(messages)}"
        model = data.get("model", "mock-vision-model")
        reasoning = canned_reasoning(last_user) if wants_reasoning(last_user) else None

        if data.get("stream"):
            self._send_stream(model, answer, reasoning)
            return

        self._send(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": data.get("model", "mock-vision-model"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": answer},
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )

    def log_message(self, fmt: str, *args) -> None:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=0,
        help="delay per streamed chunk, to watch streaming",
    )
    args = parser.parse_args()

    DELAY_MS = args.delay_ms

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(
        f"Mock llama-server listening on http://0.0.0.0:{args.port} "
        "(stateless, OpenAI-style)"
    )
    server.serve_forever()
