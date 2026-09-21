from langchain_core.chat_history import InMemoryChatMessageHistory

_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the history for a session, creating it on first access.

    RunnableWithMessageHistory calls this with the configurable session_id
    on every invocation, so sessions live and die through this accessor.
    """
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()

    return _histories[session_id]
