#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ----------------------------
# We'll reuse a structure to hold a single example's embeddings
# from the precompute script
# ----------------------------
class PrecomputedExample:
    def __init__(self, query_emb, passage_embs, label):
        """
        query_emb: shape (hidden_dim,)
        passage_embs: shape (M, hidden_dim) -- M passages total
        label: int index of which passage is originally positive
        """
        self.query_emb = query_emb
        self.passage_embs = passage_embs
        self.label = label


# ----------------------------
# A simple Dataset that wraps the loaded PrecomputedExamples
# ----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, precomputed_list):
        self.data = precomputed_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------
# Multi-class probe
# We'll treat it as a single vector:
#    [ query_emb, passage_0, passage_1, ..., passage_{N-1} ]
# => linear => N classes
# so linear in_size = hidden_dim*(N+1), out_size = N
# ----------------------------
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
        # Flatten passage_embs => (B, N*H)
        passage_flat = passage_embs.view(B, N * H)
        # Concat => (B, (N+1)*H)
        concat_vec = torch.cat([query_emb, passage_flat], dim=1)
        logits = self.linear(concat_vec)  # (B, N)
        return logits


# ----------------------------
# Collate function
#   - We sample exactly `args.num_passages` passages per example:
#       1 positive + (num_passages-1) negatives
#   - We shuffle them so the positive passage is in a random position.
# ----------------------------
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
        # Skip if not enough passages
        if full_passage_count < num_passages:
            continue

        # Pick exactly 1 positive + (num_passages-1) negatives
        pos_idx = ex.label
        pos_emb = ex.passage_embs[pos_idx].unsqueeze(0)  # shape (1, hidden_dim)

        # All negative indices
        neg_indices = [i for i in range(full_passage_count) if i != pos_idx]
        # Sample exactly (num_passages - 1) negatives
        chosen_neg_indices = random.sample(neg_indices, num_passages - 1)

        # Combine
        chosen_indices = [pos_idx] + chosen_neg_indices  # len = num_passages
        chosen_embs = ex.passage_embs[chosen_indices, :]  # shape (num_passages, hidden_dim)

        # Now shuffle them so the positive is at a random position
        perm = list(range(num_passages))
        random.shuffle(perm)
        permuted_embs = chosen_embs[perm, :]

        # Where did the first item (the original positive) end up?
        # Original positive was index=0 in chosen_embs, so new_label is
        # the index in `perm` that points to 0
        new_label = perm.index(0)

        new_batch.append((ex.query_emb, permuted_embs, new_label))

    # If nothing remains, return None so that DataLoader can skip
    if len(new_batch) == 0:
        return None

    # Build final tensors
    B = len(new_batch)
    hidden_dim = new_batch[0][0].shape[0]  # from ex.query_emb
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


def evaluate(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                # means the collate function skipped everything in this mini-batch
                continue

            q_emb = batch["q_emb"].to(device)  # (B, hidden_dim)
            p_emb = batch["p_emb"].to(device)  # (B, N, hidden_dim)
            labels = batch["labels"].to(device)  # (B,)

            logits = model(q_emb, p_emb)  # (B, N)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_folder", type=str, required=True,
                        help="Path to the folder containing chunk_*.pt for a single layer.")
    # add validation folder
    parser.add_argument("--validation_folder", type=str,default=None,
                        help="Path to the folder containing chunk_*.pt for a single layer")
    parser.add_argument("--num_passages", type=int, required=True,
                        help="Number of passages used for classification (1 positive + num_passages-1 negatives).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="multi_probe.pt",
                        help="Where to save the trained multi-class probe.")
    args = parser.parse_args()

    # Set device + seeds
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ----------------------------
    # Load all chunk_*.pt files from layer_folder
    # ----------------------------
    chunk_paths = sorted(glob.glob(os.path.join(args.layer_folder, "chunk_*.pt")))
    if not chunk_paths:
        raise ValueError(f"No chunk_*.pt files found in {args.layer_folder}")

    all_examples = []
    for cpath in chunk_paths:
        print(f"Loading {cpath}...")
        chunk_data = torch.load(cpath)  # list of PrecomputedExample
        all_examples.extend(chunk_data)

    print(f"Total loaded examples: {len(all_examples)}")
    dataset_full = EmbeddingDataset(all_examples)


    # Split train/test only if validation folder is not provided
    if (args.validation_folder is not None) and (args.validation_folder.lower() != "none") and os.path.exists(args.validation_folder):
        validation_chunk_paths = sorted(glob.glob(os.path.join(args.validation_folder, "chunk_*.pt")))
        if not validation_chunk_paths:
            raise ValueError(f"No chunk_*.pt files found in {args.validation_folder}")

        validation_examples = []
        for cpath in validation_chunk_paths:
            print(f"Loading {cpath}...")
            chunk_data = torch.load(cpath)
            validation_examples.extend(chunk_data)
        train_dataset = EmbeddingDataset(all_examples)
        test_dataset = EmbeddingDataset(validation_examples)
    else:
        train_size = int(len(dataset_full) * args.train_split)
        test_size = len(dataset_full) - train_size

        train_dataset, test_dataset = random_split(dataset_full, [train_size, test_size])

    # Build DataLoaders with a custom collate_fn that takes num_passages
    def collate_wrapper(batch):
        return collate_fn(batch, args.num_passages)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_wrapper)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_wrapper)

    # Infer hidden_dim from the first non-skipped example
    # We'll just scan until we find something with enough passages
    hidden_dim = None
    for ex in all_examples:
        if ex.passage_embs.shape[0] >= args.num_passages:
            hidden_dim = ex.query_emb.shape[0]
            break
    if hidden_dim is None:
        raise ValueError("No example found with enough passages for num_passages.")

    print(f"Initializing multi-class probe with hidden_dim={hidden_dim}, num_passages={args.num_passages}...")
    probe = MultiConcatProbe(hidden_dim, args.num_passages).to(DEVICE)

    optimizer = optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("Starting training...")
    best_acc = 0.0  # Initialize the best accuracy

    for epoch in range(args.epochs):
        probe.train()
        running_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if batch is None:
                # This batch had no valid examples
                continue

            q_emb = batch["q_emb"].to(DEVICE)  # (B, hidden_dim)
            p_emb = batch["p_emb"].to(DEVICE)  # (B, num_passages, hidden_dim)
            labels = batch["labels"].to(DEVICE)  # (B,)

            optimizer.zero_grad()
            logits = probe(q_emb, p_emb)  # (B, num_passages)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        test_acc = evaluate(test_loader, probe, DEVICE)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f} - TestAcc: {test_acc:.3f}")

        # Save the model if the test accuracy is the best we've seen so far
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best accuracy: {best_acc:.3f}. Saving model...")
            torch.save(probe.state_dict(), args.save_path)

    print("Training complete. Best model saved.")


if __name__ == "__main__":
    main()
