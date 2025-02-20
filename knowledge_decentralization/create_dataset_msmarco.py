import json
import datasets
import random
from tqdm import tqdm
import pandas as pd



def process_dataset(dataset_split):
    new_data = []
    less_count = 0
    zero_count = 0
    overall_count = 0
    for item in tqdm(dataset_split):
        question = item["query"]
        passages = item["passages"]
        passages_text= passages["passage_text"]
        labels= passages["is_selected"]

        positive_contexts = [passages_text[i] for i in range(len(labels)) if labels[i]]
        negative_contexts = [passages_text[i] for i in range(len(labels)) if not labels[i]]


        if len(negative_contexts) < 4:
            #(f"Warning: less than 4 negative contexts for question: {question}")
            #print(len(positive_contexts), len(negative_contexts))
            less_count += 1

        if len(negative_contexts) == 0:
            #print(f"Warning: no negative contexts for question: {question}")
            continue
            zero_count += 1

        for positive_context in positive_contexts:
            # sample 4 negative contexts for each positive context, but if there are less than 4 negative contexts, repeat the negative contexts

            if len(negative_contexts) < 4:
                negative_contexts_sample = random.choices(negative_contexts, k=4)
            else:
                negative_contexts_sample = random.sample(negative_contexts, k=4)
            CURRENT_ITEM = {
                "question": question,
                "positive_context": positive_context
            }
            for i, negative_context in enumerate(negative_contexts_sample):
                CURRENT_ITEM[f"negative_context_{i}"] = negative_context
            new_data.append(CURRENT_ITEM)
            overall_count += 1
    print(f"less count: {less_count}", "ratio: ", less_count/len(dataset_split))
    print(f"zero count: {zero_count}", "ratio: ", zero_count/len(dataset_split))
    print(f"overall count: {overall_count}")
    return new_data





random.seed(42)
# create a huggingface dataset using the json file
dataset_name="microsoft/ms_marco"
train_split="train"
dev_split="validation"
test_split="test"

subset="v1.1"
# first load the dataset
dataset = datasets.load_dataset(dataset_name, subset)

train_data = process_dataset(dataset["train"])
dev_data = process_dataset(dataset["validation"])
test_data = process_dataset(dataset["test"])


# create a huggingface dataset using the json file
train_dataset = datasets.Dataset.from_list(train_data)
dev_dataset = datasets.Dataset.from_list(dev_data)
test_data = datasets.Dataset.from_list(test_data)

# upload the dataset to Hugging Face with two splits
dataset_dict = datasets.DatasetDict({
    "train": train_dataset,
    "dev": dev_dataset,
    "test": test_data
})

dataset_dict.push_to_hub("wshuai190/msmarco-v1.1-hard-negative-seed-42-sample-4")

