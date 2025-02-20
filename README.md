# DenseRetriever-Knowledge-Acquisition

This repository provides a pipeline for evaluating and interpreting dense retrieval models through two main components:

1. **Knowledge Consistency Pipeline:**  
   Precomputes embeddings, trains probing models, and evaluates consistency between query/passage representations.

2. **Knowledge Decentralisation Pipeline:**  
   Computes neuron activation counts using Integrated Gradients and visualizes activation patterns.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Knowledge Consistency](#knowledge-consistency-pipeline)
  - [Knowledge Decentralisation](#knowledge-decentralisation-pipeline)
- [Environment](#environment)

---

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```bash

# Install dependencies
pip install -r environment.txt

## Usage

### Knowledge Consistency Pipeline

The pipeline for knowledge consistency consists of three stages: precomputing embeddings, training a probing model, and evaluating the probe.

#### 1. Precompute Embeddings
```bash
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

#### 2. Train the Probing Model
```bash
python train_probe_multi.py \
  --layer_folder output_precomputed/layer_0 \
  --num_passages 5 \
  --epochs 30 \
  --lr 1e-4 \
  --batch_size 32768 \
  --train_split 0.99 \
  --save_path probe_model.pt

#### 3. Evaluate the Probing Model
```bash
python evaluation.py \
  --model_path probe_model.pt \
  --data_folder output_precomputed/layer_0 \
  --num_passages 5 \
  --batch_size 32768 \
  --device cuda

