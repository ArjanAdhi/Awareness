# run.py
import sys
import os

# Add current directory to sys.path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from awareness_separate.main_agent import MainGPT

def main():
    agent = MainGPT(model_name="distilgpt2")
    print("Starting infinite monologue loop. Press Ctrl+C to stop.")
    iteration = 0

    agent.memory_manager.add_message("The agent begins its monologue in a quiet room...")

    while True:
        iteration += 1
        print(f"\n--- Iteration #{iteration} ---")
        short_notif, new_text = agent.process_iteration()
        print(short_notif)
        print("New Text:", new_text)

if __name__ == "__main__":
    main()