#!/usr/bin/env python
"""
Evaluation script for a single probing model (MultiConcatProbe) that was trained
on chunk_*.pt files containing PrecomputedExample objects.

Usage:
    python evaluation.py --model_path PATH_TO_MODEL \
                         --data_folder PATH_TO_CHUNKS \
                         --num_passages 2 \
                         [--batch_size BATCH_SIZE] \
                         [--device DEVICE]

This script loads the model checkpoint from model_path, loads all chunk_*.pt
files in data_folder (each a list of PrecomputedExample objects), builds a DataLoader,
runs inference, and prints the accuracy.
"""

import argparse
import glob
import os
import random
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#from torch.serialization import safe_globals

# ----------------------------------------------------------
# 1) Define a PrecomputedExample class (matching your training code).
#    Each chunk_*.pt is a list of these objects.
# ----------------------------------------------------------
class PrecomputedExample:
    """
    Holds the [CLS] embeddings for one example (for a specific layer):
      - query_emb: shape (hidden_dim,)
      - passage_embs: shape (num_passages, hidden_dim)
      - label: int (the index of the positive passage, typically 0)
    """
    def __init__(self, query_emb, passage_embs, label):
        self.query_emb = query_emb
        self.passage_embs = passage_embs
        self.label = label


# ----------------------------------------------------------
# 2) Wrap them in a Dataset (matching your training code).
# ----------------------------------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, precomputed_list):
        self.data = precomputed_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------------------------------------
# 3) We need the same collate_fn that your training used.
#    This function randomly shuffles the passages, but for *evaluation*
#    you can also leave it as-is if you want a random arrangement each time.
#    (Alternatively, you can remove the shuffle for a deterministic eval.)
# ----------------------------------------------------------
def collate_fn(batch, num_passages):
    """
    batch: list of PrecomputedExample objects
    We'll produce:
      - q_emb: (B, hidden_dim)
      - p_emb: (B, num_passages, hidden_dim)
      - labels: (B,) with values in [0..num_passages-1] indicating
                which passage is the 'positive' one
    """
    new_batch = []
    for ex in batch:
        full_passage_count = ex.passage_embs.shape[0]
        if full_passage_count < num_passages:
            # skip if not enough passages
            continue

        pos_idx = ex.label
        pos_emb = ex.passage_embs[pos_idx].unsqueeze(0)

        # All negative indices
        neg_indices = [i for i in range(full_passage_count) if i != pos_idx]
        # Sample exactly (num_passages - 1) negatives
        chosen_neg_indices = random.sample(neg_indices, num_passages - 1)

        # Combine
        chosen_indices = [pos_idx] + chosen_neg_indices
        chosen_embs = ex.passage_embs[chosen_indices, :]

        # Shuffle them so the positive is at a random position
        perm = list(range(num_passages))
        random.shuffle(perm)
        permuted_embs = chosen_embs[perm, :]

        new_label = perm.index(0)  # where did the original positive end up?

        new_batch.append((ex.query_emb, permuted_embs, new_label))

    if len(new_batch) == 0:
        return None

    # Build final tensors
    B = len(new_batch)
    hidden_dim = new_batch[0][0].shape[0]
    q_list, p_list, lbl_list = [], [], []

    for (q_emb, p_emb, lbl) in new_batch:
        q_list.append(q_emb)
        p_list.append(p_emb)
        lbl_list.append(lbl)

    q_tensor = torch.stack(q_list)  # (B, hidden_dim)
    p_tensor = torch.stack(p_list)  # (B, num_passages, hidden_dim)
    labels = torch.tensor(lbl_list, dtype=torch.long)  # (B,)

    return {
        "q_emb": q_tensor,
        "p_emb": p_tensor,
        "labels": labels
    }


# ----------------------------------------------------------
# 4) Evaluate function, same as in train_probe_multi.py
# ----------------------------------------------------------
def evaluate(loader, model, device):
    model.eval()
    correct_list = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            q_emb = batch["q_emb"].to(device)
            p_emb = batch["p_emb"].to(device)
            labels = batch["labels"].to(device)


            logits = model(q_emb, p_emb)  # (B, num_passages)
            preds = torch.argmax(logits, dim=-1)
            # correct_list append 1 for correct, 0 for incorrect
            correct_list.extend((preds == labels).cpu().numpy().tolist())
    return correct_list


# ----------------------------------------------------------
# 5) MultiConcatProbe (from train_probe_multi.py)
# ----------------------------------------------------------
class MultiConcatProbe(nn.Module):
    def __init__(self, hidden_dim, num_passages):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_passages = num_passages
        # in_features = hidden_dim*(num_passages + 1), out_features = num_passages
        self.linear = nn.Linear(hidden_dim * (num_passages + 1), num_passages)

    def forward(self, query_emb, passage_embs):
        """
        query_emb: (B, hidden_dim)
        passage_embs: (B, N, hidden_dim) where N = self.num_passages
        Returns: (B, N) classification logits over the N passages
        """
        B, N, H = passage_embs.shape
        passage_flat = passage_embs.view(B, N * H)  # (B, N*H)
        concat_vec = torch.cat([query_emb, passage_flat], dim=1)  # (B, (N+1)*H)
        logits = self.linear(concat_vec)  # (B, N)
        return logits


# ----------------------------------------------------------
# 6) Main: load model & data, run evaluation
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a MultiConcatProbe model using chunk_*.pt files.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model's state_dict.")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Folder containing chunk_*.pt files.")
    parser.add_argument("--num_passages", type=int, required=True,
                        help="Number of passages used in training (the model input dimension).")
    parser.add_argument("--batch_size", type=int, default=32768,
                        help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on (cuda, cpu, or mps).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set device and seeds
    output_dir = os.path.dirname(args.model_path)
    output_path = os.path.join(output_dir, "evaluation_results.txt")
    print(f"Saving accuracy to {output_path}")
    # if os.path.exists(output_path):
    #     # if it is not empty
    #     with open(output_path, "r") as f:
    #         if f.read().strip():
    #             return  # already evaluated
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)




    # 6a) Load the trained model
    model_state = torch.load(args.model_path, map_location=device)
    model = MultiConcatProbe(hidden_dim=int((model_state["linear.weight"].size()[1]/(args.num_passages+1))), num_passages=args.num_passages)
    model.load_state_dict(model_state)
    model.to(device)

    # 6b) Load all chunk_*.pt into a single dataset
    import_fn_pattern = os.path.join(args.data_folder, "chunk_*.pt")
    chunk_paths = sorted(glob.glob(import_fn_pattern))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.pt files found in {args.data_folder}")

    all_examples = []
    for cpath in tqdm(chunk_paths):
        chunk_data = torch.load(cpath, weights_only=False)  # list of PrecomputedExample
        all_examples.extend(chunk_data)

    dataset = EmbeddingDataset(all_examples)

    # 6c) Build DataLoader using the same collate_fn
    def collate_wrapper(batch):
        return collate_fn(batch, args.num_passages)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper)

    # 6d) Evaluate and print accuracy
    accuracy_list = evaluate(dataloader, model, device)

    print(f"Accuracy: {sum(accuracy_list) / len(accuracy_list):.4f}")
    with open(output_path, "w") as f:
        for acc in accuracy_list:
            f.write(f"{acc}\n")



if __name__ == "__main__":
    main()
