import random

import numpy as np
import regex
from datasets import Dataset

from src.helpers import compute_pass_at_k


class FinetuningDataset:
    def __init__(
        self,
        seed,
        tokenizer,
        apply_chat_template,
        k=4,
        train_size=50000,
        val_size=5000,
        test_id_size=10000,
        test_ood_size=10000,
    ):
        output_pattern = r"""\\boxed\{
            (?P<content>
                (?:
                    [^{}]
                | (?P<brace>
                        \{
                            (?: [^{}] | (?&brace) )*
                        \}
                    )
                )*
            )
        \}"""
        self.output_pattern = output_pattern
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template
        self.seed = seed
        self.k = k
        self.train_size = train_size
        self.val_size = val_size
        self.test_id_size = test_id_size
        self.test_ood_size = test_ood_size
        self.min_val = 10 ** (k - 1)  # e.g., 1000 for k=4
        self.max_val = 10**k - 1  # e.g., 9999 for k=4

    def extract_output(self, completion: str):
        last = None
        for m in regex.finditer(
            self.output_pattern, completion, flags=regex.DOTALL | regex.VERBOSE
        ):
            last = m.group("content")
        return last

    def extract_output_first(self, completion: str):
        m = regex.search(
            self.output_pattern, completion, flags=regex.DOTALL | regex.VERBOSE
        )
        if m:
            return m.group("content")
        return None

    def eval_outputs(self, outputs, pass_at_k: int, is_base_model: bool, scratch):
        accuracy = 0
        accuracy_normalized = 0
        pass_at_k_scores = {k: 0 for k in range(1, pass_at_k + 1)}
        all_predictions = []

        def normalize_whitespace(text):
            """Remove all whitespace from text."""
            if text is None:
                return ""
            return "".join(text.split())

        for i, out in enumerate(outputs):
            print(f"Evaluating output {i + 1}/{len(outputs)}")

            # Get the predictions
            texts = [t for t in out["generated_text"]]
            example_id = out["example_id"]
            ground_truth = out["ground_truth"]
            preds = []
            num_correct = 0
            num_correct_normalized = 0

            # Get ground truth (last element if list, otherwise the value)
            gt = ground_truth[-1] if isinstance(ground_truth, list) else ground_truth
            gt_normalized = normalize_whitespace(gt)

            for text in texts:
                if not scratch:
                    if is_base_model:
                        pred = self.extract_output_first(text)
                    else:
                        pred = self.extract_output(text)
                else:
                    pred = text.strip()
                preds.append(pred)

                # Exact match
                if pred is not None and pred == gt:
                    num_correct += 1

                # Whitespace-normalized match
                pred_normalized = normalize_whitespace(pred)
                if pred_normalized == gt_normalized:
                    num_correct_normalized += 1

            example_accuracy = num_correct / len(texts) if len(texts) > 0 else 0
            example_accuracy_normalized = (
                num_correct_normalized / len(texts) if len(texts) > 0 else 0
            )
            accuracy += example_accuracy
            accuracy_normalized += example_accuracy_normalized

            current_pass_at_k = {k: 0 for k in range(1, pass_at_k + 1)}
            for k in range(1, pass_at_k + 1):
                p_at_k = compute_pass_at_k(len(texts), num_correct_normalized, k)
                pass_at_k_scores[k] += p_at_k
                current_pass_at_k[k] = p_at_k
            all_predictions.append(
                {
                    "example_id": example_id,
                    "ground_truth": ground_truth,
                    "accuracy": example_accuracy,
                    "accuracy_normalized": example_accuracy_normalized,
                    "pass_at_k": {k: p_at_k for k, p_at_k in current_pass_at_k.items()},
                    "predictions": preds,
                    "prompt": out["prompt"],
                    "texts": texts,
                }
            )
        accuracy /= len(outputs)
        accuracy_normalized /= len(outputs)
        for k in pass_at_k_scores:
            pass_at_k_scores[k] /= len(outputs)
        return accuracy, accuracy_normalized, pass_at_k_scores, all_predictions

    def generate_data(self):
        """Generate addition problems with non-overlapping train/val/test splits."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Generate all unique problem pairs
        all_pairs = set()
        max_attempts = (self.max_val - self.min_val + 1) ** 2
        attempts = 0

        # Generate enough unique pairs for all splits
        total_needed = self.train_size + self.val_size + self.test_id_size
        while len(all_pairs) < total_needed and attempts < max_attempts:
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(self.min_val, self.max_val)
            all_pairs.add((a, b))
            attempts += 1

        # Convert to list and shuffle deterministically
        all_pairs = list(all_pairs)
        random.shuffle(all_pairs)

        # Split deterministically
        train_pairs = all_pairs[: self.train_size]
        val_pairs = all_pairs[self.train_size : self.train_size + self.val_size]
        test_pairs = all_pairs[
            self.train_size + self.val_size : self.train_size
            + self.val_size
            + self.test_id_size
        ]

        # Generate examples
        def create_example(a, b):
            """Create a single addition example."""
            result = a + b
            prompt = f"{a} + {b} ="
            completion = str(result)
            return {
                "prompt": prompt,
                "completion": completion,
                "a": a,
                "b": b,
                "result": result,
            }

        train_examples = [create_example(a, b) for a, b in train_pairs]
        val_examples = [create_example(a, b) for a, b in val_pairs]
        test_examples = [create_example(a, b) for a, b in test_pairs]

        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        test_dataset = Dataset.from_list(test_examples)

        return train_dataset, val_dataset, test_dataset

    def generate_ood_data(self, exclude_pairs=None):
        """Generate OOD data with thousands separator format.

        Args:
            exclude_pairs: Set of (a, b) pairs to exclude (from train/val/test)
        """
        if exclude_pairs is None:
            exclude_pairs = set()

        random.seed(self.seed + 999)  # Different seed for OOD
        np.random.seed(self.seed + 999)

        # Generate unique pairs not in exclude_pairs
        ood_pairs = []
        max_attempts = (self.max_val - self.min_val + 1) ** 2
        attempts = 0

        while len(ood_pairs) < self.test_ood_size and attempts < max_attempts:
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(self.min_val, self.max_val)
            if (a, b) not in exclude_pairs:
                ood_pairs.append((a, b))
            attempts += 1

        def format_with_separator(num):
            """Format number with thousands separator (space)."""
            s = str(num)
            # Add space every 3 digits from right
            result = []
            for i, char in enumerate(reversed(s)):
                if i > 0 and i % 3 == 0:
                    result.append(" ")
                result.append(char)
            return "".join(reversed(result))

        def create_ood_example(a, b):
            """Create OOD example with thousands separator in input."""
            result = a + b
            # Format input with separators
            a_formatted = format_with_separator(a)
            b_formatted = format_with_separator(b)
            prompt = f"{a_formatted} + {b_formatted}="
            completion = str(
                result
            )  # Output without separator (but accept any spacing)
            return {
                "prompt": prompt,
                "completion": completion,
                "a": a,
                "b": b,
                "result": result,
            }

        ood_examples = [create_ood_example(a, b) for a, b in ood_pairs]
        ood_dataset = Dataset.from_list(ood_examples)

        return ood_dataset


def get_datasets(args, tokenizer):
    """Get train and validation datasets for training.

    Args:
        args: Hydra config object
        tokenizer: Tokenizer instance

    Returns:
        train_dataset, eval_dataset (HuggingFace Dataset objects)
    """
    dataset_pars = args.dataset_pars
    k = getattr(dataset_pars, "k", 4)
    train_size = getattr(dataset_pars, "train_size", 50000)
    val_size = getattr(dataset_pars, "val_size", 5000)
    test_id_size = getattr(dataset_pars, "test_id_size", 10000)
    test_ood_size = getattr(dataset_pars, "test_ood_size", 10000)

    dataset = FinetuningDataset(
        seed=args.seed,
        tokenizer=tokenizer,
        apply_chat_template=dataset_pars.apply_chat_template,
        k=k,
        train_size=train_size,
        val_size=val_size,
        test_id_size=test_id_size,
        test_ood_size=test_ood_size,
    )

    train_dataset, val_dataset, _ = dataset.generate_data()

    return train_dataset, val_dataset


def get_dataset(args, tokenizer, split="test_id"):
    """Get test dataset for inference/evaluation.

    Args:
        args: Hydra config object
        tokenizer: Tokenizer instance
        split: 'test_id' or 'test_ood'

    Returns:
        test_dataset (HuggingFace Dataset object)
    """
    dataset_pars = args.dataset_pars
    k = getattr(dataset_pars, "k", 4)
    train_size = getattr(dataset_pars, "train_size", 50000)
    val_size = getattr(dataset_pars, "val_size", 5000)
    test_id_size = getattr(dataset_pars, "test_id_size", 10000)
    test_ood_size = getattr(dataset_pars, "test_ood_size", 10000)

    dataset = FinetuningDataset(
        seed=args.seed,
        tokenizer=tokenizer,
        apply_chat_template=dataset_pars.apply_chat_template,
        k=k,
        train_size=train_size,
        val_size=val_size,
        test_id_size=test_id_size,
        test_ood_size=test_ood_size,
    )

    if split == "test_id":
        _, _, test_dataset = dataset.generate_data()
        return test_dataset
    elif split == "test_ood":
        # Generate ID data first to get exclude pairs
        train_dataset, val_dataset, test_id_dataset = dataset.generate_data()
        # Collect all pairs from ID splits
        exclude_pairs = set()
        for ds in [train_dataset, val_dataset, test_id_dataset]:
            for ex in ds:
                exclude_pairs.add((ex["a"], ex["b"]))
        ood_dataset = dataset.generate_ood_data(exclude_pairs=exclude_pairs)
        return ood_dataset
    else:
        raise ValueError(f"Unknown split: {split}. Use 'test_id' or 'test_ood'")
