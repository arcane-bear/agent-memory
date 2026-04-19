"""Quickstart: use agent-memory as a library."""

from agent_memory import MemoryStore


def main() -> None:
    store = MemoryStore("./demo.db")

    # Store some memories.
    store.add("the user prefers dark mode in all apps", metadata={"source": "chat"})
    store.add("the user is a vegetarian", metadata={"source": "profile"})
    store.add("the user lives in berlin and speaks english and german")
    store.add("the user's cat is named moxie", ttl=7 * 86400)  # forget in a week

    # Recall by similarity.
    results = store.search("what theme does the user like?", limit=2)
    for r in results:
        print(f"[{r.score:.3f}] {r.memory.content}")

    print(f"\nTotal memories: {store.count()}")
    print(f"Namespaces: {store.namespaces()}")


if __name__ == "__main__":
    main()
