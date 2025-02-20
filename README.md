Below is the complete README file along with an environment.txt file. You can copy and paste these into your repository.

# DenseRetriever-Knowledge-Acquisition

This repository provides a pipeline for evaluating and interpreting dense retrieval models through two main components:

1. **Knowledge Consistency Pipeline:**  
   This component precomputes query and passage embeddings, trains a probing model to assess the consistency between query and passage representations, and evaluates the probe’s performance.

2. **Knowledge Decentralisation (Neuron Attribution) Pipeline:**  
   This component computes per-neuron activation counts using Integrated Gradients (IG) on the dense layers of transformer models. It also provides a plotting utility to visualize neuron activation heatmaps, which help interpret which neurons are most “active” based on IG thresholds.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Knowledge Consistency Pipeline](#knowledge-consistency-pipeline)
    - [Precompute Embeddings](#1-precompute-embeddings)
    - [Train the Probing Model](#2-train-the-probing-model)
    - [Evaluate the Probing Model](#3-evaluate-the-probing-model)
  - [Knowledge Decentralisation Pipeline](#knowledge-decentralisation-pipeline)
    - [Compute Neuron Activation Counts](#1-compute-neuron-activation-counts)
    - [Plot Activation Heatmap](#2-plot-activation-heatmap)
- [Environment](#environment)

---

## Installation

Create a virtual environment (optional but recommended) and install the required packages. See [environment.txt](environment.txt) for the dependency list.

```bash
# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r environment.txt
Usage

Knowledge Consistency Pipeline
The pipeline for knowledge consistency consists of three stages: precomputing embeddings, training a probing model, and evaluating the probe.

1. Precompute Embeddings

Use the provided script (e.g., precompute_embeddings.py) to compute and store embeddings for queries and passages. The script supports PEFT models if required.

python precompute_embeddings.py \
  --model_name bert-base-uncased \
  --dataset_name tomaarsen/natural-questions-hard-negatives \
  --dataset_config default \
  --dataset_split train \
  --chunk_size 100 \
  --batch_size 32 \
  --query_max_length 128 \
  --passage_max_length 512 \
  --encoder_type cls \
  --output_dir output_precomputed \
  --use_peft False \
  --save_original True
This command processes the dataset in chunks (each containing 100 examples), computes embeddings (using the CLS token or the specified method), and saves the results under output_precomputed/ organized by layer (e.g., layer_0/chunk_*.pt).

2. Train the Probing Model

Train the multi-class probe (implemented in train_probe_multi.py) to learn which passage (among a positive and negatives) best matches the query representation.

python train_probe_multi.py \
  --layer_folder output_precomputed/layer_0 \
  --num_passages 5 \
  --epochs 30 \
  --lr 1e-4 \
  --batch_size 32768 \
  --train_split 0.99 \
  --save_path probe_model.pt
Here, the probe is trained on embeddings from layer 0 with 5 passages per example (1 positive and 4 negatives). The best model is saved as probe_model.pt.

3. Evaluate the Probing Model

After training, evaluate the probe using the evaluation script (evaluation.py). This script loads the trained probe and all the precomputed chunk files for a given layer.

python evaluation.py \
  --model_path probe_model.pt \
  --data_folder output_precomputed/layer_0 \
  --num_passages 5 \
  --batch_size 32768 \
  --device cuda
The evaluation script computes the accuracy of the probe and saves detailed results in an output file.

Knowledge Decentralisation Pipeline
This pipeline computes per-neuron activation counts using Integrated Gradients and then visualizes the results.

1. Compute Neuron Activation Counts

The script (e.g., compute_attributions.py) computes Integrated Gradients (IG) for the intermediate and output dense layers of a transformer. For each example, it normalizes the IG values by the global maximum across layers and counts neurons whose IG exceeds one or more specified thresholds.

python compute_attributions.py \
  --model bert-base-uncased \
  --encoder_type CLS \
  --alpha 20 \
  --batch_size 8 \
  --dataset wshuai190/nq-hard-negative-dpr-sampled-seed-42-sample-4 \
  --max_example 1000 \
  --save_folder attributions \
  --thresholds 0.1 0.2 \
  --num_chunks 5 \
  --chunk_index 0 \
  --target positive_context
This command processes a chunk of the dataset (if chunking is desired) and saves the IG counts for each threshold to JSON files under attributions/threshold_<thr>/chunk_<chunk_index>.json.

2. Plot Activation Heatmap

After computing attributions, use the plotting script (e.g., plot_activations.py) to visualize the aggregated neuron activations as a heatmap. The script aggregates activation counts into bins and produces a side-by-side heatmap for different attribution folders.

python plot_activations.py \
  --save_folders attributions/threshold_0.1 attributions/threshold_0.2 \
  --num_bins 10 \
  --max_number 54414
This command creates a PDF file (e.g., under a graphs folder) showing the activation heatmaps.

Environment

An environment.txt file is provided for easy installation of all required dependencies. See below.

Additional Information

PEFT Support:
The code optionally supports loading and merging LoRA/PEFT models. If you wish to use PEFT, ensure that the PEFT package is installed.
Model Architectures:
The scripts have been tested with models such as bert-base-uncased but should work with other transformer models that follow similar architectures.
Dataset Requirements:
The scripts utilize the Hugging Face Datasets library. Ensure that the datasets you choose have the expected fields (e.g., query, positive_context, etc.).
Feel free to adjust parameters and paths as needed.

Happy experimenting and interpreting!


And here is the `environment.txt` file:

```txt
# environment.txt
torch>=1.10.0
transformers>=4.0.0
datasets>=1.6.0
tqdm>=4.0.0
numpy>=1.18.0
matplotlib>=3.0.0
peft>=0.2.0   # Optional, only needed if using PEFT models
These files should provide a comprehensive guide to set up, run, and interpret the knowledge consistency and decentralisation pipelines in your repository. Enjoy!
