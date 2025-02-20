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

# Install dependencies
pip install -r environment.txt
