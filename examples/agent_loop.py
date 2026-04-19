"""Example: wire agent-memory into a chat agent loop.

This is framework-agnostic — it shows the pattern of retrieving relevant
memories before each LLM call and storing new facts after. Drop in
whichever LLM client you prefer.
"""

from agent_memory import MemoryStore


def build_context(store: MemoryStore, user_message: str, *, k: int = 5) -> str:
    """Retrieve the top-k memories relevant to the current turn."""
    hits = store.search(user_message, limit=k, min_score=0.1)
    if not hits:
        return ""
    lines = [f"- {h.memory.content}" for h in hits]
    return "Relevant memories:\n" + "\n".join(lines)


def remember(store: MemoryStore, fact: str, *, session_id: str) -> None:
    """Store a fact scoped to a single session/agent."""
    store.add(fact, namespace=session_id, metadata={"kind": "fact"})


def main() -> None:
    store = MemoryStore("./agent.db", namespace="session-123")

    # Seed a few facts.
    remember(store, "user's name is Ada", session_id="session-123")
    remember(store, "user works on a cryptography project in Rust", session_id="session-123")
    remember(store, "user prefers concise explanations", session_id="session-123")

    turn = "Remind me what I'm working on."
    ctx = build_context(store, turn)
    print("--- prompt context ---")
    print(ctx)
    print("--- user turn ---")
    print(turn)


if __name__ == "__main__":
    main()
