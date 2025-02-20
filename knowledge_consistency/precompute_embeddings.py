#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import torch
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from peft import PeftModel, PeftConfig
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_name, use_peft=False):
    if use_peft and PeftConfig is not None:
        config = PeftConfig.from_pretrained(model_name)
        base_model_name = config.base_model_name_or_path
        print(f"Detected PEFT model. Loading base model: {base_model_name}")
        base_model = AutoModel.from_pretrained(
            base_model_name,
            output_hidden_states=True
        ).to(DEVICE)
        model = PeftModel.from_pretrained(
            base_model,
            model_name
        )
        model = model.merge_and_unload()
        print("Successfully loaded PEFT model.")
    else:
        print("PEFT loading disabled or not applicable. Falling back to standard AutoModel loading.")
        model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
    model.to(DEVICE)
    model.eval()
    return model

def get_tokenizer(model_name, use_peft=False):
    if use_peft and PeftConfig is not None:
        config = PeftConfig.from_pretrained(model_name)
        base_model_name = config.base_model_name_or_path
        print(f"Detected PEFT config. Loading tokenizer from base model: {base_model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
        )
    else:
        print("Tokenizer loading via PEFT config disabled or not applicable. Falling back to using the model name directly.")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
    # if model does not have pad token, put pad token as eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def compute_embeddings(model, input_ids, attention_mask, encoder_type, num_layers=13):
    """
    Runs the model in no_grad mode and returns a list of embeddings for each layer,
    computed according to the chosen encoder_type:
      - "cls": use the CLS token (index 0).
      - "mean": compute mean pooling over tokens (using the attention mask).
      - "contriever": compute mean pooling (using the attention mask) and then apply L2 normalization.
    Returns a list of length num_layers, each of shape (batch_size, hidden_dim).
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple with length = num_layers
        layers_emb = []
        for layer_idx in range(num_layers):
            layer_hidden = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_dim)
            if encoder_type.lower() == "cls":
                emb = layer_hidden[:, 0, :]
            elif encoder_type.lower() == "mean":
                mask = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
                summed = torch.sum(layer_hidden * mask, dim=1)
                counts = mask.sum(dim=1)
                counts[counts == 0] = 1  # avoid division by zero
                emb = summed / counts
            elif encoder_type.lower() == "eos":
                # Directly use the last token embedding (assuming the EOS token is last)
                emb = layer_hidden[:, -1, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            else:
                emb = layer_hidden[:, 0, :]
            layers_emb.append(emb.cpu())
    return layers_emb


def batched_forward(model, input_ids, attention_mask, num_layers, batch_size, encoder_type):
    # we want a list of length num_layers,
    # each an empty list that we'll fill with embeddings.
    all_layers_acc = [ [] for _ in range(num_layers) ]

    total = input_ids.size(0)
    for start_idx in tqdm(range(0, total, batch_size), desc="Forward Batches"):
        end_idx = min(start_idx + batch_size, total)
        batch_ids = input_ids[start_idx:end_idx].to(DEVICE)
        batch_mask = attention_mask[start_idx:end_idx].to(DEVICE)

        layer_representation_list = compute_embeddings(model, batch_ids, batch_mask, encoder_type, num_layers)
        # layer_cls_list is length num_layers, each shape (bsize, hidden_dim)
        for l_idx in range(num_layers):
            all_layers_acc[l_idx].append(layer_representation_list[l_idx])
    # Now concatenate for each layer
    for l_idx in range(num_layers):
        all_layers_acc[l_idx] = torch.cat(all_layers_acc[l_idx], dim=0)  # (total, hidden_dim)
    return all_layers_acc

def process_chunk(chunk_data, chunk_idx, tokenizer, base_out_dir, encoder, query_encoder= None, query_tokenizer=None, batch_size=256, num_layers=13, query_max_length=128, passage_max_length=512, encoder_type="cls", save_original=False):
    """
    chunk_data: list of (query_text, [passages], label)
    chunk_idx: int index of the chunk

    We'll:
      1) tokenize queries
      2) tokenize passages
      3) run in mini-batches to get hidden_states
      4) organize them by example
      5) for each layer, create PrecomputedExample objects
      6) save them in the appropriate directory
    """
    print(f"Processing chunk #{chunk_idx} with {len(chunk_data)} examples...")

    # Separate out query_texts and passage_texts (flattened)
    queries = []
    passages_flat = []
    passage_counts = []
    for (q, p_list, lbl) in chunk_data:
        queries.append(q)
        passage_counts.append(len(p_list))
        for p in p_list:
            passages_flat.append(p)
    if query_encoder is None:
        query_encoder = encoder

    # 1) Tokenize queries
    if encoder_type=="eos":
        queries = ["query: " + q + " </s>" for q in queries]
        passages_flat = ["passage: " + p + " </s>" for p in passages_flat]


    if query_tokenizer is None:
        q_enc = tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=query_max_length,
            return_tensors="pt"
        )
    else:
        q_enc = query_tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=query_max_length,
            return_tensors="pt"
        )

    # 2) Tokenize passages
    p_enc = tokenizer(
        passages_flat,
        padding="max_length",
        truncation=True,
        max_length=passage_max_length,
        return_tensors="pt"
    )

    # 3) Compute CLS in mini-batches for queries

    print(f"Computing {encoder_type} for queries...")
    query_layers = batched_forward(query_encoder, q_enc["input_ids"], q_enc["attention_mask"], num_layers,  batch_size=batch_size, encoder_type=encoder_type)
    # query_layers is a list of length num_layers,
    # each shape => (num_examples_in_chunk, hidden_dim).

    print(f"Computing {encoder_type} for passages...")
    passage_layers = batched_forward(encoder, p_enc["input_ids"], p_enc["attention_mask"], num_layers, batch_size=batch_size, encoder_type=encoder_type)
    # passage_layers is a list of length num_layers,
    # each shape => (sum_of_passage_counts, hidden_dim).

    # 4) Now reorganize the passage CLS by example
    # We'll do a pointer (p_idx) to step through passage_layers
    p_idx = 0
    # For each layer, we want to build a list of PrecomputedExample
    # We'll store them in a dictionary: layer -> list_of_examples
    layer_to_examples = { l_idx: [] for l_idx in range(num_layers) }

    for i, (q_text, p_list, lbl) in enumerate(chunk_data):
        n_passages = passage_counts[i]
        for l_idx in range(num_layers):
            original_q = q_text
            positive_context = p_list[0]
            negative_contexts = p_list[1:]
            q_emb = query_layers[l_idx][i]  # shape (hidden_dim,)
            p_embs = passage_layers[l_idx][p_idx : p_idx + n_passages]  # (num_passages, hidden_dim)
            if save_original:
                example_obj = PrecomputedExample(q_emb, p_embs, lbl, original_q, positive_context, negative_contexts)
            else:
                example_obj = PrecomputedExample(q_emb, p_embs, lbl)
            layer_to_examples[l_idx].append(example_obj)
        p_idx += n_passages

    # 5) Save them in the appropriate directory:
    # We will have directories like:
    #   {base_out_dir}/layer_0/
    #   {base_out_dir}/layer_1/
    #   ...
    # And inside each, we save chunk_{chunk_idx}.pt
    for l_idx in range(num_layers):
        layer_dir = os.path.join(base_out_dir, f"layer_{l_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        out_path = os.path.join(layer_dir, f"chunk_{chunk_idx}.pt")
        torch.save(layer_to_examples[l_idx], out_path)
        # Freed from memory after saving
    print(f"Finished chunk #{chunk_idx}.")

# ----------------------------
# A simple class to hold the data
# ----------------------------
class PrecomputedExample:
    """
    Holds the [CLS] embeddings for one example (for a specific layer):
      - query_emb: shape (hidden_dim,)
      - passage_embs: shape (num_passages, hidden_dim)
      - label: int (the index of the positive passage, typically 0)
    """
    def __init__(self, query_emb, passage_embs, label, question=None, positive_context=None, negative_contexts=None):
        self.query_emb = query_emb
        self.passage_embs = passage_embs
        self.label = label
        if question is not None:
            self.question = question
        if positive_context is not None:
            self.positive_context = positive_context
        if negative_contexts is not None:
            self.negative_contexts = negative_contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    # add query encoder seperate name if it is not None
    parser.add_argument("--query_encoder_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="tomaarsen/natural-questions-hard-negatives")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Number of examples per chunk file.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for the forward pass.")
    parser.add_argument("--query_max_length", type=int, default=128)
    parser.add_argument("--chunk_index", type=int, default=None)
    parser.add_argument("--passage_max_length", type=int, default=512)
    parser.add_argument("--num_passages", type=int, default=5)  # 1 positive + 4 negatives
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output_precomputed",
                        help="Directory to save the precomputed embeddings.")
    parser.add_argument("--encoder_type", type=str, default="cls",
                        help="Encoding method: 'cls', 'mean', or 'contriever'")
    # NEW FLAG for PEFT
    parser.add_argument("--use_peft", type=bool, default=False,
                        help="If set, attempt to load and merge LoRA/PEFT models.")
    parser.add_argument("--save_original", type=bool, default=True)
    args = parser.parse_args()
    print(args)
    # Set device and seeds
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)


    if args.query_encoder_name:
        print(f"Query Encoder: {args.query_encoder_name}")

    print(f"Device: {DEVICE}")

    # ----------------------------
    # Load dataset
    # ----------------------------
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    if args.max_examples > 0:
        raw_dataset = raw_dataset.select(range(min(args.max_examples, len(raw_dataset))))

    # ----------------------------
    # Load model + tokenizer
    # (frozen, no gradient)
    # ----------------------------

    # Load tokenizer + model
    tokenizer = get_tokenizer(args.model_name, use_peft=args.use_peft)
    model = get_model(args.model_name, use_peft=args.use_peft)

    query_tokenizer = None
    query_encoder = None
    if args.query_encoder_name is not None:
        query_tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_name, use_fast=True)
        query_encoder = AutoModel.from_pretrained(args.query_encoder_name, output_hidden_states=True).to(DEVICE)
        query_encoder.eval()

    # check if the model have seperate query encoder

    model.eval()


    for param in model.parameters():
        param.requires_grad = False

    # Number of layers to extract is model.config.num_hidden_layers + 1 if
    # we include the initial embedding layer in hidden_states[0].
    # Typically for BERT-base: hidden_states indices = 0..12
    #  - 0 = embeddings, 1..12 = each of the 12 transformer layers
    num_layers = model.config.num_hidden_layers + 1

    # ----------------------------
    # Prepare queries and passages
    # ----------------------------
    # We'll flatten queries+passages just like before, but do it in chunks.
    # We'll store the entire dataset text to avoid reloading text.
    # But we won't encode them all at once; we do it chunk by chunk.
    all_data = []
    for idx in range(len(raw_dataset)):
        example = raw_dataset[idx]
        if "query" in example:
            query_text = example["query"]
        elif "question" in example:
            query_text = example["question"]
        if "answer" in example:
            pos_passage = example["answer"]
        elif "correct_answer" in example:
            pos_passage = example["correct_answer"]
        elif "positive_context" in example:
            pos_passage = example["positive_context"]

        if "incorrect_answers" in example:
            negs = [example["incorrect_answers"]]
        elif "positive_context" in example:
            negs = [example.get(f"negative_context_{i}", "") for i in range(0, args.num_passages-1)]
        else:
            negs = [example.get(f"negative_{i}", "") for i in range(1, args.num_passages)]
        # If not enough negatives, skip
        if len(negs) < (args.num_passages - 1):
            continue
        passages = [pos_passage] + negs
        label = 0  # first is positive
        all_data.append((query_text, passages, label))

    print(f"Total examples after filtering: {len(all_data)}")

    # Create output directory:
    # e.g. {output_dir}/{dataset_name}_{model_name}
    if args.query_encoder_name:
        base_out_dir = os.path.join(args.output_dir, f"{args.dataset_name}_{args.dataset_split}_{args.model_name}_{args.encoder_type}_{args.query_encoder_name.split('/')[-1]}")
    else:
        base_out_dir = os.path.join(args.output_dir, f"{args.dataset_name}_{args.dataset_split}_{args.model_name}_{args.encoder_type}")

    os.makedirs(base_out_dir, exist_ok=True)
    print(f"Base output directory: {base_out_dir}")


    # Break up all_data into chunk_size
    total_data = len(all_data)
    chunk_count = (total_data + args.chunk_size - 1) // args.chunk_size
    if args.chunk_index is not None:
        chunk_index = args.chunk_index
        start = chunk_index * args.chunk_size
        end = min(start + args.chunk_size, total_data)
        chunk_data = all_data[start:end]
        process_chunk(chunk_data, chunk_index, tokenizer, base_out_dir, model, query_encoder, query_tokenizer, args.batch_size, num_layers=num_layers, query_max_length=args.query_max_length, passage_max_length=args.passage_max_length, encoder_type=args.encoder_type, save_original=args.save_original)
    else:
        for chunk_idx in range(chunk_count):
            start = chunk_idx * args.chunk_size
            end = min(start + args.chunk_size, total_data)
            chunk_data = all_data[start:end]
            process_chunk(chunk_data, chunk_idx, tokenizer, base_out_dir, model, query_encoder, query_tokenizer, args.batch_size, num_layers=num_layers, query_max_length=args.query_max_length, passage_max_length=args.passage_max_length, encoder_type=args.encoder_type, save_original=args.save_original)

    print("All chunks processed. Done.")

if __name__ == "__main__":
    main()
