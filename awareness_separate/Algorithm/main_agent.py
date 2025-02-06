# main_gpt.py

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments

from .subconscious_gpt import SubconsciousGPT
from .memory_manager import MemoryManager
from .node_manager import NodeManager
from .weight_master import WeightMaster


class MainGPT:
    def __init__(self, model_name="distilgpt2"):
        # Model & tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to("cpu")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

        self.subconscious = SubconsciousGPT()
        self.memory_manager = MemoryManager()
        self.node_manager = NodeManager()
        self.weight_master = WeightMaster()

        self.iteration_count = 0
        self.fine_tune_interval = 10

    def conscious_prune_if_needed(self, prompt_ids, max_new_tokens=50):
        """
        Conscious pruning if token length would exceed GPT-2 limit (1024).
        We'll keep only the last portion of the prompt to fit (1024 - max_new_tokens).
        """
        max_input_len = 1024 - max_new_tokens
        if prompt_ids.size(1) > max_input_len:
            # We truncate from the front
            truncated_part = prompt_ids[:, :-max_input_len]
            prompt_ids = prompt_ids[:, -max_input_len:]

            # Convert truncated tokens to text for carried context
            truncated_text = self.tokenizer.decode(truncated_part[0], skip_special_tokens=True)
            print("[Conscious] On-the-fly pruning performed to avoid token overflow.")
            print(f"[Conscious] Carried truncated text snippet: {truncated_text[:50]}...")

        return prompt_ids

    def process_iteration(self):
        """
        1. Subconscious global prune
        2. Build prompt
        3. Conscious prune if needed
        4. Generate
        5. Possibly fine-tune
        """
        self.iteration_count += 1

        # 1. Subconscious global prune (keep last 20 lines, for example)
        history_list = self.memory_manager.get_history_list()
        new_list = self.subconscious.prune_memory_global(history_list, max_messages=20)
        self.memory_manager.set_history_list(new_list)

        # 2. Build prompt from memory
        full_context = self.memory_manager.get_full_context()
        sub_stats = self.subconscious.analyze_context(full_context)
        node_out = self.node_manager.process_with_nodes(full_context)

        prompt = (
            f"{full_context}\n\n"
            f"Subconscious analysis: {sub_stats}\n"
            f"Node outputs: {node_out}\n"
            "Continuing monologue:\n"
        )

        # 3. Conscious prune if needed
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = self.conscious_prune_if_needed(input_ids, max_new_tokens=50)

        # 4. Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract the newly generated portion
        new_text = generated_text[len(prompt):].strip() or generated_text.strip()
        self.memory_manager.add_message(new_text)

        # Incoherence measure
        incoherence = 0.0
        if len(new_text) < 10:
            incoherence = 0.5
        elif new_text in full_context:
            incoherence = 0.3

        self.weight_master.update_parameters(incoherence)

        do_finetune = self.weight_master.should_finetune()
        if self.iteration_count % self.fine_tune_interval == 0:
            do_finetune = True

        if do_finetune:
            self.fine_tune_self()
            self.subconscious.fine_tune_self(full_context)

        short_notif = (
            f"Awareness: {self.weight_master.awareness:.2f} | "
            f"Curiosity: {self.weight_master.curiosity:.2f} | "
            f"Happiness: {self.weight_master.happiness:.2f} | "
            f"Tune: {'yes' if do_finetune else 'no'}"
        )
        return short_notif, new_text

    def fine_tune_self(self):
        """
        Simple example of on-the-fly fine tuning for the main GPT model.
        """
        memory_text = self.memory_manager.get_full_context()
        if len(memory_text) < 20:
            print("[MainGPT] Not enough memory to fine-tune. Skipping.")
            return

        print("[MainGPT] Fine-tuning on conversation memory...")

        # Build dataset
        tokens = self.tokenizer(memory_text, return_tensors="pt", max_length=128, truncation=True)
        input_ids = tokens["input_ids"]
        attn_mask = tokens["attention_mask"]

        # Minimal Trainer approach
        training_args = TrainingArguments(
            output_dir="training",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=10,
            logging_steps=10,
            logging_dir="training/logs",
            do_eval=False,
        )

        def data_collator(features):
            batch = {
                "input_ids": features[0]["input_ids"],
                "attention_mask": features[0]["attention_mask"],
                "labels": features[0]["input_ids"]
            }
            return batch

        from torch.utils.data import Dataset

        class MemoryDataset(Dataset):
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return {
                    "input_ids": input_ids[0],
                    "attention_mask": attn_mask[0]
                }

        from transformers import Trainer
        trainer = Trainer(
            model=GPT2LMHeadModel.from_pretrained("distilgpt2"),
            args=training_args,
            train_dataset=MemoryDataset(),
            data_collator=data_collator,
        )
        trainer.train()

        checkpoint_path = f"training/checkpoint-iteration-{self.iteration_count}"
        trainer.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        print(f"[MainGPT] Checkpoint saved to {checkpoint_path}.")