# subconscious_gpt.py

class SubconsciousGPT:
    """
    Handles behind-the-scenes tasks like global pruning and optional fine-tuning.
    """

    def __init__(self):
        self.carried_context = []  # store truncated contexts for reference

    def analyze_context(self, conversation_context: str) -> dict:
        """
        Return stats about the entire conversation context.
        """
        words = conversation_context.split()
        char_count = sum(len(w) for w in words)
        return {
            "word_count": len(words),
            "char_count": char_count
        }

    def prune_memory_global(self, conversation_history: list, max_messages=20) -> list:
        """
        Prune conversation if it exceeds `max_messages`.
        Keep the latest messages, store truncated portion in `carried_context`.
        """
        if len(conversation_history) > max_messages:
            # We'll keep the last `max_messages` lines, remove older lines
            to_prune = conversation_history[:-max_messages]
            kept = conversation_history[-max_messages:]

            # Store pruned lines in carried_context for reference
            self.carried_context.append("\n".join(to_prune))

            # Provide a notification
            print("[Subconscious] Global pruning performed. "
                  f"Removed {len(to_prune)} lines from memory.\n"
                  f"Carried context size: {len(self.carried_context)} segments.")

            return kept
        return conversation_history

    def fine_tune_self(self, memory_data: str):
        """
        Placeholder for subconscious fine-tuning.
        """
        print("[SubconsciousGPT] Fine-tuning self on memory data (placeholder).")