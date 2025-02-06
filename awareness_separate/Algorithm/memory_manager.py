# memory_manager.py

class MemoryManager:
    """
    Stores conversation messages in a simple list.
    """

    def __init__(self):
        self.conversation_history = []

    def add_message(self, msg: str):
        self.conversation_history.append(msg)

    def get_history_list(self):
        return self.conversation_history

    def set_history_list(self, new_list):
        self.conversation_history = new_list

    def get_full_context(self):
        """
        Return the entire conversation as one string.
        """
        return "\n".join(self.conversation_history)