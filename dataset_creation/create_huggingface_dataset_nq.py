import json
import datasets
import random
from tqdm import tqdm
import pandas as pd

def process_file(train_file):
    with open(train_file, "r") as f:
        train_data = json.load(f)

    new_train_data = []
    for item in tqdm(train_data):
        question = item["question"]
        positive_contexts = item["positive_ctxs"]

        negative_contexts = item["hard_negative_ctxs"]
        if len(negative_contexts) < 4:
            print(f"Warning: less than 4 negative contexts for question: {question}")

        for positive_context in positive_contexts:
            # sample 4 negative contexts for each positive context, but if there are less than 4 negative contexts, repeat the negative contexts
            if len(negative_contexts) == 0:
                print(f"Warning: no negative contexts for question: {question}")
                continue
            if len(negative_contexts) < 4:
                negative_contexts_sample = random.choices(negative_contexts, k=4)
            else:
                negative_contexts_sample = random.sample(negative_contexts, k=4)
            CURRENT_ITEM = {
                "question": question,
                "positive_context": positive_context['title'] + " " + positive_context['text']
            }
            for i, negative_context in enumerate(negative_contexts_sample):
                CURRENT_ITEM[f"negative_context_{i}"] = negative_context['title'] + " " + negative_context['text']
            new_train_data.append(CURRENT_ITEM)
    return new_train_data

random.seed(42)
# create a huggingface dataset using the json file
train_file = "DPR-main/dpr/downloads/data/retriever/nq-adv-hn-train.json"
dev_file = "DPR-main/dpr/downloads/data/retriever/nq-dev.json"

train_data = process_file(train_file)
dev_data = process_file(dev_file)

# create a huggingface dataset using the json file
train_dataset = datasets.Dataset.from_list(train_data)
dev_dataset = datasets.Dataset.from_list(dev_data)

# save the dataset
train_dataset.save_to_disk("DPR-main/dpr/downloads/data/retriever/nq-adv-hn-train")
dev_dataset.save_to_disk("DPR-main/dpr/downloads/data/retriever/nq-adv-hn-dev")

# upload the dataset to Hugging Face with two splits
dataset_dict = datasets.DatasetDict({
    "train": train_dataset,
    "dev": dev_dataset
})

dataset_dict.push_to_hub("wshuai190/nq-hard-negative-dpr-sampled-seed-42-sample-4")

