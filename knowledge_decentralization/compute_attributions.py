#!/usr/bin/env python
# coding=utf-8
"""
Compute and store per-neuron activation counts using integrated gradients.
For each example, we compute integrated gradients (IG) attributions for the dense
layers (both intermediate and output) in each transformer layer. Then for each specified
threshold, we count whether a neuron's IG (normalized relative to the maximum IG across
all layers for that example) exceeds that threshold. The final output is saved to one file
per threshold (e.g. attributions_0.1.json).
"""

import argparse
import json
import os
import torch
import numpy as np
import time
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import math

# Attempt to import PEFT classes if available.
try:
    from peft import PeftModel, PeftConfig
except ImportError:
    PeftModel, PeftConfig = None, None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def get_model(model_name, use_peft=False):
    """
    Dynamically load a model.
    If --use_peft is set (and PEFT is installed), load the PEFT config and then the base model.
    After calling `merge_and_unload()`, we re-enable requires_grad on all parameters
    so that integrated gradients can flow through the merged weights.
    """
    if use_peft and PeftConfig is not None:
        config = PeftConfig.from_pretrained(model_name)
        base_model_name = config.base_model_name_or_path
        print(f"Detected PEFT model. Loading base model: {base_model_name}")
        base_model = AutoModel.from_pretrained(
            base_model_name, output_hidden_states=False
        ).to(DEVICE)

        # Load the LoRA model on top of the base
        model = PeftModel.from_pretrained(base_model, model_name)

        # Merge LoRA weights into the base model
        model = model.merge_and_unload()

        # Re-enable gradient on merged parameters
        for param in model.parameters():
            param.requires_grad_(True)

        print("Successfully loaded PEFT model (with merged weights). "
              "Re-enabled 'requires_grad' on parameters.")
    else:
        print("Loading standard AutoModel.")
        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=False
        ).to(DEVICE)

    model.eval()
    return model


def get_tokenizer(model_name, use_peft=False):
    """
    Dynamically load a tokenizer.
    If --use_peft is set (and PEFT is installed), load the tokenizer from the base model.
    """
    if use_peft and PeftConfig is not None:
        config = PeftConfig.from_pretrained(model_name)
        base_model_name = config.base_model_name_or_path
        print(f"Detected PEFT config. Loading tokenizer from base model: {base_model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    else:
        print("Loading tokenizer using model name.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        print("No pad_token found. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def preprocess_texts(texts, encoder_type, target="positive_context"):
    """
    If encoder_type is 'eos', preprocess texts by adding a prefix ("passage: ")
    and appending an EOS marker ("</s>"). Otherwise, return texts unchanged.
    """
    if encoder_type.lower() == "eos":
        if target == "positive_context":
            return [f"passage: {text}</s>" for text in texts]
        elif target == "question":
            return [f"query: {text}</s>" for text in texts]
    return texts


def get_model_layers(model):
    """
    Extract the list of transformer layers from the model.
    Supports various architectures.
    """
    if hasattr(model, "encoder"):
        return model.encoder.layer
    elif hasattr(model, "ctx_encoder"):
        return model.ctx_encoder.bert_model.encoder.layer
    elif hasattr(model, "question_encoder"):
        return model.question_encoder.bert_model.encoder.layer
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    else:
        raise ValueError("Unsupported model architecture for layer extraction.")


def compute_batch_layerwise_global_ig_counts(model, dense_list, inputs, n_steps=20, encoder_type="cls",
                                             thresholds=[0.1]):
    """
    For each example in the batch, compute integrated gradients (IG) for each dense layer
    separately but normalize all neuron IG values by the global maximum across all layers.

    For each example:
      1. Compute per-layer integrated gradients (accumulated over integration steps).
      2. For each layer, sum the absolute IG over input dimensions to get per-neuron IG.
      3. Concatenate all neuron IGs across layers to compute the global maximum.
      4. Normalize each layer’s neuron IG by the global max.
      5. For each threshold, count the neuron as "active" if its normalized IG > threshold.

    Returns a dictionary mapping each threshold to a list of count arrays (one per dense layer)
    for this batch.
    """
    batch_size = inputs[list(inputs.keys())[0]].shape[0]

    # Save original parameters for each dense layer.
    original_params = []
    for layer in dense_list:
        W_orig = layer.weight.data.clone()
        b_orig = layer.bias.data.clone() if layer.bias is not None else None
        original_params.append((layer, W_orig, b_orig))

    # Prepare accumulators for counts per threshold, per dense layer.
    batch_counts = {thr: [np.zeros(layer.weight.shape[0], dtype=np.int64) for layer in dense_list]
                    for thr in thresholds}

    eps = 1e-6
    alpha_values = torch.linspace(eps, 1.0, steps=n_steps, device=DEVICE)
    delta_alpha = 1.0 / n_steps

    # Process each example individually.
    for ex in range(batch_size):
        # List to store per-layer integrated gradients (IG) for this example.
        per_layer_igs = []

        # Compute IG for each dense layer individually.
        for (layer, W_orig, b_orig) in original_params:
            accumulator = torch.zeros_like(W_orig, device=DEVICE)
            for alpha in alpha_values:
                # Scale only this layer's parameters.
                layer.weight.data = W_orig * alpha
                if b_orig is not None:
                    layer.bias.data = b_orig * alpha

                # Prepare single-example inputs.
                single_inputs = {k: v[ex:ex + 1] for k, v in inputs.items()}
                with torch.enable_grad():
                    outputs = model(**single_inputs)
                    # Choose the token representation.
                    if hasattr(outputs, "pooler_output") and encoder_type.lower() == "cls":
                        token_rep = outputs.pooler_output
                    else:
                        hidden = outputs.last_hidden_state
                        if encoder_type.lower() == "mean":
                            token_rep = hidden.mean(dim=1)
                        elif encoder_type.lower() == "eos":
                            token_rep = hidden[:, -1, :]
                        else:
                            token_rep = hidden[:, 0, :]
                    out_scalar = token_rep.norm()

                # Compute gradients for the current layer.
                grad = torch.autograd.grad(out_scalar, layer.weight, retain_graph=True)[0]
                accumulator += grad * (W_orig * delta_alpha)

            # Restore the original parameters.
            layer.weight.data = W_orig
            if b_orig is not None:
                layer.bias.data = b_orig

            # Compute the per-neuron IG by summing over the input dimensions.
            neuron_ig = accumulator.abs().sum(dim=1)  # Shape: [num_neurons]
            per_layer_igs.append(neuron_ig)

        # Compute the global maximum IG over all layers for this example.
        all_neuron_igs = torch.cat(per_layer_igs)
        global_max = all_neuron_igs.max().item()
        if global_max == 0:
            global_max = 1.0  # Prevent division by zero.

        # For each layer, normalize by the global max and update counts.
        for idx, neuron_ig in enumerate(per_layer_igs):
            norm_ig = neuron_ig / global_max
            for thr in thresholds:
                active = (norm_ig.detach().cpu().numpy() > thr)
                batch_counts[thr][idx] += active.astype(np.int64)

    return batch_counts


def compute_batch_ig_counts(model, dense_list, inputs, n_steps=20, encoder_type="cls", thresholds=[0.1]):
    """
    For each example in the batch, compute integrated gradients (IG) for each dense layer
    using the all-at-once approach.

    For each example, we:
      1. Compute the IG for each dense layer (accumulated over integration steps).
      2. Collect all per-neuron IG values across **all** dense layers and compute the global maximum.
      3. Normalize each neuron's IG by this global maximum.
      4. For each threshold, count the neuron as "active" if its normalized IG > threshold.

    Returns a dictionary mapping each threshold to a list of count arrays (one per dense layer)
    for this batch.
    """
    batch_size = inputs[list(inputs.keys())[0]].shape[0]

    # Save original parameters for each dense layer.
    original_params = []
    for layer in dense_list:
        W_orig = layer.weight.data.clone()
        b_orig = None
        if layer.bias is not None:
            b_orig = layer.bias.data.clone()
        original_params.append((layer, W_orig, b_orig))

    # Prepare accumulators for counts per threshold, per dense layer.
    batch_counts = {thr: [np.zeros(layer.weight.shape[0], dtype=np.int64) for layer in dense_list]
                    for thr in thresholds}

    eps = 1e-6
    alpha_values = torch.linspace(eps, 1.0, steps=n_steps, device=DEVICE)
    delta_alpha = 1.0 / n_steps

    # Process each example individually.
    for ex in range(batch_size):
        # For each dense layer, initialize an accumulator (shape: out_dim x in_dim)
        accumulators = [torch.zeros_like(p[1], device=DEVICE) for p in original_params]

        # Loop over integration steps.
        for alpha in alpha_values:
            # Scale each dense layer's weights by alpha.
            for (layer_module, W_orig, b_orig) in original_params:
                layer_module.weight.data = W_orig * alpha
                if b_orig is not None:
                    layer_module.bias.data = b_orig * alpha

            # Prepare single-example inputs.
            single_inputs = {k: v[ex:ex + 1] for k, v in inputs.items()}
            with torch.enable_grad():
                outputs = model(**single_inputs)
                # Decide how to pick the representation.
                if hasattr(outputs, "pooler_output") and encoder_type.lower() == "cls":
                    token_rep = outputs.pooler_output
                else:
                    hidden = outputs.last_hidden_state  # fallback for BERT-style models
                    if encoder_type.lower() == "mean":
                        token_rep = hidden.mean(dim=1)
                    elif encoder_type.lower() == "eos":
                        token_rep = hidden[:, -1, :]
                    else:
                        token_rep = hidden[:, 0, :]
                # Compute a scalar from the token representation.
                out_scalar = token_rep.norm()

            # Compute gradients for each dense layer's weight.
            grads = torch.autograd.grad(
                out_scalar, [p[0].weight for p in original_params],
                retain_graph=True
            )

            # Accumulate integrated gradients for this example.
            for idx, grad in enumerate(grads):
                accumulators[idx] += grad * (original_params[idx][1] * delta_alpha)

        # Restore original parameters for each dense layer.
        for (layer_module, W_orig, b_orig) in original_params:
            layer_module.weight.data = W_orig
            if b_orig is not None:
                layer_module.bias.data = b_orig

        # For each dense layer, compute per-neuron integrated gradient by summing over input dims.
        # We first collect all the per-neuron IG values across layers to compute a global maximum.
        all_neuron_igs = []
        per_layer_igs = []
        for accumulator in accumulators:
            neuron_ig = accumulator.abs().sum(dim=1)  # shape: [out_dim]
            per_layer_igs.append(neuron_ig)
            all_neuron_igs.append(neuron_ig)

        # Concatenate all values to get the global max.
        global_neuron_igs = torch.cat(all_neuron_igs)
        global_max = global_neuron_igs.max().item()

        # If global_max is 0, then skip normalization (all values are 0)
        if global_max > 0:
            # For each layer, normalize and update counts for each threshold.
            for idx, neuron_ig in enumerate(per_layer_igs):
                norm_ig = neuron_ig / global_max
                for thr in thresholds:
                    active = (norm_ig.detach().cpu().numpy() > thr)
                    batch_counts[thr][idx] += active.astype(np.int64)
        else:
            # Global max is zero; update counts with zeros.
            pass

    return batch_counts


def collect_activations(model, tokenizer, texts, batch_size, n_steps, encoder_type, thresholds, target="positive_context"):
    """
    Process texts in batches, computing counts of per-neuron activations (i.e., number
    of examples for which the neuron's normalized IG, relative to the global maximum across
  all layers for that example, exceeds the given thresholds) for each neuron's weights in
    each layer's intermediate and output dense layers.
    """
    texts = preprocess_texts(texts, encoder_type, target=target)

    layers = get_model_layers(model)
    dense_list = []
    layer_keys = []

    # Build a list of "intermediate" and "output" dense modules.
    for i, layer in enumerate(layers):
        if hasattr(layer, 'mlp'):  # e.g. LLaMA-style
            inter_linear = layer.mlp.gate_proj
            out_linear = layer.mlp.down_proj
        else:  # BERT / DPR style
            inter_linear = layer.intermediate.dense
            out_linear = layer.output.dense

        dense_list.append(inter_linear)
        layer_keys.append((i, "intermediate"))
        dense_list.append(out_linear)
        layer_keys.append((i, "output"))

    # Initialize global count arrays for each threshold.
    global_counts = {thr: [] for thr in thresholds}
    for linear in dense_list:
        out_dim = linear.weight.shape[0]
        for thr in thresholds:
            global_counts[thr].append(np.zeros(out_dim, dtype=np.int64))

    num_batches = int(np.ceil(len(texts) / batch_size))
    progress_bar = tqdm(range(0, len(texts), batch_size),
                        desc="Computing IG counts",
                        total=num_batches,
                        unit="batch")

    for idx in progress_bar:
        start_time = time.time()
        batch_texts = texts[idx: idx + batch_size]

        # Tokenize.
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Compute IG counts for this batch (for each threshold).
        batch_counts = compute_batch_layerwise_global_ig_counts(
            model, dense_list, inputs,
            n_steps=n_steps,
            encoder_type=encoder_type,
            thresholds=thresholds
        )
        # Update global counts: for each threshold, for each dense layer, add the counts.
        for thr in thresholds:
            for i, count_arr in enumerate(batch_counts[thr]):
                global_counts[thr][i] += count_arr

        batch_time = time.time() - start_time
        progress_bar.set_postfix({'batch_time': f'{batch_time:.2f}s'})

    # Convert global counts to dictionary format:
    # { "layer_0_intermediate": count_array, "layer_0_output": count_array, ... } for each threshold.
    final_dict = {}
    for thr in thresholds:
        layer_dict = {}
        for i, (layer_idx, layer_type) in enumerate(layer_keys):
            key = f"layer_{layer_idx}_{layer_type}"
            layer_dict[key] = global_counts[thr][i].tolist()
        final_dict[thr] = layer_dict
    return final_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and store per-neuron activation counts using integrated gradients."
    )
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model name or path (default: bert-base-uncased)")
    parser.add_argument("--encoder_type", type=str, default="CLS",
                        help="Target token representation (CLS, mean, eos) (default: CLS)")
    parser.add_argument("--alpha", type=int, default=20,
                        help="Number of integration steps (default: 20)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing texts (default: 8)")
    parser.add_argument("--dataset", type=str,
                        default="wshuai190/nq-hard-negative-dpr-sampled-seed-42-sample-4",
                        help="Dataset name (default: Tevatron/wikipedia-nq-corpus)")
    parser.add_argument("--max_example", type=int, default=None,
                        help="Maximum number of examples to process (default: all)")
    parser.add_argument("--save_folder", type=str, default="attributions",
                        help="Base file path (without threshold extension) to save attributions")
    parser.add_argument("--thresholds", type=float, nargs='+', default=[0.1],
                        help="One or more threshold values (default: [0.1])")
    # New arguments for chunking.
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Total number of chunks to split the dataset into.")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="Which chunk index to process (0-based).")
    parser.add_argument("--target", type=str, default="positive_context",
                        help="Which field to use as the target text.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Arguments:", args)

    # 1. Load dataset and optionally select a chunk.
    dataset = load_dataset(args.dataset, split="dev")



    texts = []
    for i, example in enumerate(tqdm(dataset, desc="Reading dataset")):
        if args.max_example is not None and i >= args.max_example:
            break
        texts.append(example[args.target])

    #deduplicate texts but do not change order

    texts = list(dict.fromkeys(texts))

    total_length = len(texts)
    print(f"Total dataset length: {total_length}")

    if args.num_chunks is not None and args.chunk_index is not None:
        chunk_size = math.ceil(total_length / args.num_chunks)
        start_idx = chunk_size * args.chunk_index
        end_idx = min(start_idx + chunk_size, total_length)
        print(f"Processing chunk {args.chunk_index}/{args.num_chunks} "
              f"covering indices [{start_idx}:{end_idx}).")
        texts = texts[start_idx:end_idx]
    else:
        print("Processing entire dataset (no chunk splitting).")

    # 2. Gather texts from the dataset chunk.


    print(f"Collected {len(texts)} passages from chunk.")

    # 3. Load model and tokenizer; compute integrated gradients activation counts.
    try:
        model = get_model(args.model, use_peft=False)
        tokenizer = get_tokenizer(args.model, use_peft=False)
    except:
        model = get_model(args.model, use_peft=True)
        tokenizer = get_tokenizer(args.model, use_peft=True)



    print("Computing integrated gradients activation counts...")
    # global_counts is a dict mapping threshold -> { layer_key: count_array }
    global_counts = collect_activations(
        model, tokenizer, texts,
        batch_size=args.batch_size,
        n_steps=args.alpha,
        encoder_type=args.encoder_type,
        thresholds=args.thresholds,
        target=args.target
    )

    # 4. Save results to disk (one file per threshold).
    os.makedirs(args.save_folder, exist_ok=True)

    for thr, attr_dict in global_counts.items():
        # Build a filename that appends the threshold value, e.g., attributions_0.1.json
        threshold_folder = os.path.join(args.save_folder, f"threshold_{thr}")
        os.makedirs(threshold_folder, exist_ok=True)
        save_path = os.path.join(threshold_folder, f"chunk_{args.chunk_index}.json")
        output = {
            "model": args.model,
            "encoder_type": args.encoder_type,
            "alpha": args.alpha,
            "threshold": thr,
            "chunk_index": args.chunk_index,
            "num_chunks": args.num_chunks,
            "attributions": attr_dict
        }
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Attributions for threshold {thr} saved to {save_path}")


if __name__ == "__main__":
    main()
