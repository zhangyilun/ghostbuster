"""
Convert raid 500k OOD split data + logprobs into ghostbuster format.
Using "id" as the file name instead of enumeration.
"""
import os
import json
from tqdm.auto import tqdm

import pandas as pd
from transformers import AutoTokenizer

# empty ids
exclude_ids = set([
    "a2dce006-20d4-455b-948d-035be6e6678d"
])

MAX_LEN = 512
model_name = "falcon-7b"

data_folder = "data/RAID_500k__OOD/"
logits_file = "data/raid_500k.logits.json"

with open(logits_file) as f:
    logits = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

# read train
for d in os.listdir(data_folder):
    print(f"Processing train {d=}")

    train_file = os.path.join(data_folder, d, "train.with_dev.balanced.csv")
    df = pd.read_csv(train_file)

    human_folder = f"data/raid_500k_{d}/train/human"
    human_logprob_folder = os.path.join(human_folder, "logprobs")
    gpt_folder = f"data/raid_500k_{d}/train/gpt"
    gpt_logprob_folder = os.path.join(gpt_folder, "logprobs")
    os.makedirs(human_logprob_folder, exist_ok=True)
    os.makedirs(gpt_logprob_folder, exist_ok=True)
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        _id = row["id"]
        text = row["text"]
        if _id in exclude_ids: continue
        if len(text.split()) < 2: continue 

        tokenized = tokenizer.tokenize(text, truncation=True, max_length=MAX_LEN)[1:] # list[str]
        _logit = logits[_id]["logits1"][0] # list[float]
    
        # determine output file names
        if row.model == "human":
            outfile = os.path.join(human_folder, f"{_id}.txt")
            logprobfile = os.path.join(human_logprob_folder, f"{_id}-{model_name}.txt")
        else:
            outfile = os.path.join(gpt_folder, f"{_id}.txt")
            logprobfile = os.path.join(gpt_logprob_folder, f"{_id}-{model_name}.txt")        
    
        # save text
        with open(outfile, "wb") as f:
            f.write(text.encode())
        
        # save logprobs
        to_write = ""
        for w, p in zip(tokenized, _logit):
            to_write += f"{w} {p}\n"
        with open(logprobfile, "w") as f:
            f.write(to_write)


# write test files
for d in os.listdir(data_folder):
    print(f"Processing test {d=}")
    test_file = os.path.join(data_folder, d, "test.csv")
    df = pd.read_csv(test_file)
    
    human_folder = f"data/raid_500k_{d}/test/human"
    human_logprob_folder = os.path.join(human_folder, "logprobs")
    gpt_folder = f"data/raid_500k_{d}/test/gpt"
    gpt_logprob_folder = os.path.join(gpt_folder, "logprobs")
    os.makedirs(human_logprob_folder, exist_ok=True)
    os.makedirs(gpt_logprob_folder, exist_ok=True)
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        _id = row["id"]
        if _id in exclude_ids: continue
        text = row["text"]
        tokenized = tokenizer.tokenize(text, truncation=True, max_length=MAX_LEN)[1:] # list[str]
        _logit = logits[_id]["logits1"][0] # list[float]
    
        # determine output file names
        if row.model == "human":
            outfile = os.path.join(human_folder, f"{_id}.txt")
            logprobfile = os.path.join(human_logprob_folder, f"{_id}-{model_name}.txt")
        else:
            outfile = os.path.join(gpt_folder, f"{_id}.txt")
            logprobfile = os.path.join(gpt_logprob_folder, f"{_id}-{model_name}.txt")        
    
        # save text
        try:
            with open(outfile, "wb") as f:
                f.write(text.encode())
        except Exception as e:
            print("Error", outfile, e)
            print(text)
        
        # save logprobs
        to_write = ""
        for w, p in zip(tokenized, _logit):
            to_write += f"{w} {p}\n"
        with open(logprobfile, "w") as f:
            f.write(to_write)
