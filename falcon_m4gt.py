import os
import sys
import argparse

from utils.featurize import convert_file_to_logprob_file, get_logprobs
from utils.load import Dataset, get_generate_dataset
from utils.n_gram import TrigramBackoff
from utils.featurize import select_features, normalize
from utils.symbolic import vec_functions, scalar_functions

from transformers import AutoTokenizer
from collections import defaultdict

import math
import tqdm
import pickle
import numpy as np
import pandas as pd

from nltk.util import ngrams
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


results_folder = "results_m4gt"
os.makedirs(results_folder, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, required=True)
parser.add_argument("--feature_select", action="store_true")
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--num_sample", type=int, default=6000, help="Number of samples to use for training if --do_sample")

parser.add_argument("--classify", action="store_true")
parser.add_argument("--features", type=str, help="File containing features line by line")
args = parser.parse_args()

domains = [ args.domain ]

train_datasets = [
    Dataset("normal", f"data/m4gt_{d}/train/{t}")
    for d in domains for t in ["human", "gpt"]
]
test_datasets = [
    Dataset("normal", f"data/m4gt_{d}/test/{t}")
    for d in domains for t in ["human", "gpt"]
]


models = ["gpt"]
vectors = ["falcon-logprobs", "unigram-logprobs", "trigram-logprobs"]

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

print("Constructing trigram")
sentences = brown.sents()

tokenized_corpus = []
for sentence in tqdm.tqdm(sentences):
    tokens = tokenizer(" ".join(sentence))["input_ids"]
    tokenized_corpus += tokens

trigram = TrigramBackoff(tokenized_corpus)


vec_combinations = defaultdict(list)
for vec1 in range(len(vectors)):
    for vec2 in range(vec1):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations[vectors[vec1]].append(f"{func} {vectors[vec2]}")

for vec1 in vectors:
    for vec2 in vectors:
        if vec1 != vec2:
            vec_combinations[vec1].append(f"v-div {vec2}")


def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")


def backtrack_functions(
    max_depth=2,
):
    """
    Backtrack all possible features.
    """

    def helper(prev, depth):
        if depth >= max_depth:
            return []

        all_funcs = []
        prev_word = get_words(prev)[-1]

        for func in scalar_functions:
            all_funcs.append(f"{prev} {func}")

        for comb in vec_combinations[prev_word]:
            all_funcs += helper(f"{prev} {comb}", depth + 1)

        return all_funcs

    ret = []
    for vec in vectors:
        ret += helper(vec, 0)
    return ret


def score_ngram(doc, model, tokenizer, n=3):
    """
    Returns vector of ngram probabilities given document, model and tokenizer
    """
    scores = []
    tokens = (
        tokenizer(doc)[1:] if n == 1 else (n - 2) * [2] + tokenizer(doc)
    )

    for i in ngrams(tokens, n):
        scores.append(model.n_gram_probability(i))

    return np.array(scores)


def get_all_logprobs(
    generate_dataset,
    preprocess=lambda x: x,
    verbose=True,
    trigram=None,
    tokenizer=None,
    split=None,
    num_tokens=511, # max_length-1
):
    falcon_logprobs = {}
    trigram_logprobs, unigram_logprobs = {}, {}

    if verbose:
        print(f"Loading logprobs into memory, {split=}")

    file_names = generate_dataset(lambda file: file, verbose=False, split=split)
    to_iter = tqdm.tqdm(file_names) if verbose else file_names

    for file in to_iter:
        if "logprobs" in file:
            continue

        with open(file, "rb") as f:
            doc = preprocess(f.read().decode())
        falcon_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "falcon-7b") # ***
        )[:num_tokens]
        trigram_logprobs[file] = score_ngram(doc, trigram, tokenizer, n=3)[:num_tokens]
        unigram_logprobs[file] = score_ngram(doc, trigram.base, tokenizer, n=1)[
            :num_tokens
        ]

    return falcon_logprobs, trigram_logprobs, unigram_logprobs


all_funcs = backtrack_functions(max_depth=3)
print(f"{len(all_funcs)=}")


# train and test datasets, labels and indices
# since we separated train and test during loading, indices are all in each category
train_generate_dataset_fn = get_generate_dataset(*train_datasets)
test_generate_dataset_fn = get_generate_dataset(*test_datasets)
train_labels = train_generate_dataset_fn(
    lambda file: 1 if "gpt" in file else 0
)
test_labels = test_generate_dataset_fn(
    lambda file: 1 if "gpt" in file else 0
)

# get ids for saving purpose
test_ids = test_generate_dataset_fn(
    lambda file: file.split("/")[-1][:-4]
)

# indices, but we use all anyways
train = np.arange(train_labels.size)
test = np.arange(test_labels.size)

# sample
sample_size = args.num_sample
if args.do_sample:
    np.random.seed(0)
    train = np.random.choice(train, size=sample_size, replace=False)
    train_labels = np.array(train_labels)[train]
print(f"{len(train)=}, {len(train_labels)=}, {len(test)=}")


# Construct all indices
def get_indices(filter_fn):
    where = np.where(train_generate_dataset_fn(filter_fn))[0]

    curr_train = [i for i in train if i in where]
    curr_test = [i for i in test if i in where]

    return curr_train, curr_test


indices_dict = {}

for model in models + ["human"]:
    train_indices, test_indices = get_indices(
        lambda file: 1 if model in file else 0,
    )

    indices_dict[f"{model}_train"] = train_indices
    indices_dict[f"{model}_test"] = test_indices


for model in models + ["human"]:
    for domain in domains:
        train_key = f"{model}_{domain}_train"
        test_key = f"{model}_{domain}_test"

        train_indices, test_indices = get_indices(
            lambda file: 1 if domain in file and model in file else 0,
        )

        indices_dict[train_key] = train_indices
        indices_dict[test_key] = test_indices

print(f"{indices_dict.keys()=}")
# human_train, human_test, gpt_train, gpt_test
# human_{d}_train, human_{d}_test, gpt_{d}_train, gpt_{d}_test


if args.feature_select:
    (
        train_falcon_logprobs,
        train_trigram_logprobs,
        train_unigram_logprobs
    ) = get_all_logprobs(
        train_generate_dataset_fn,
        verbose=True,
        split=train,
        tokenizer=lambda x: tokenizer(x).input_ids,
        trigram=trigram
    )

    train_vector_map = {
        "falcon-logprobs": lambda file: train_falcon_logprobs[file],
        "trigram-logprobs": lambda file: train_trigram_logprobs[file],
        "unigram-logprobs": lambda file: train_unigram_logprobs[file],
    }

    def calc_features(file, exp):
        exp_tokens = get_words(exp)
        curr = train_vector_map[exp_tokens[0]](file)

        for i in range(1, len(exp_tokens)):
            try:
                if exp_tokens[i] in vec_functions:
                    next_vec = train_vector_map[exp_tokens[i + 1]](file)
                    curr = vec_functions[exp_tokens[i]](curr, next_vec)
                elif exp_tokens[i] in scalar_functions:
                    return scalar_functions[exp_tokens[i]](curr)
            except Exception as e:
                print(e)
                print(exp_tokens, i, exp_tokens[i])
                print("ERROR", file, exp)
                sys.exit(1)

    print("Preparing exp_to_data")
    exp_to_data = {}
    for exp in tqdm.tqdm(all_funcs):
        exp_to_data[exp] = train_generate_dataset_fn(
            lambda file: calc_features(file, exp),
            split=train
        ).reshape(-1, 1)

    best_features = select_features(exp_to_data, train_labels, verbose=True, to_normalize=True, indices=None) # None here since we already filtered the data
    print(best_features)
    
    # save best features
    outfolder_name = f"best_features_{args.domain}"
    if args.do_sample:
        outfolder_name += "_sample"
    outfolder_name += ".txt"
    with open(os.path.join(results_folder, outfolder_name), "w") as f:
        for feat in best_features:
            f.write(feat + "\n")


if args.classify:
    if not args.features:
        raise("You must provide --features <fpath> with file containing features to use.")
        sys.exit(1)

    print(f"Evaluating domain: {args.domain}")
    print(f"Features to use: {args.features}")
    
    with open(args.features) as f:
        best_features = f.readlines()
        best_features = [l.strip() for l in best_features] # get rid of tailing \n
    print(best_features)

    (
        train_falcon_logprobs,
        train_trigram_logprobs,
        train_unigram_logprobs
    ) = get_all_logprobs(
        train_generate_dataset_fn,
        verbose=True,
        split=train,
        tokenizer=lambda x: tokenizer(x).input_ids,
        trigram=trigram
    )
    (
        test_falcon_logprobs,
        test_trigram_logprobs,
        test_unigram_logprobs
    ) = get_all_logprobs(
        test_generate_dataset_fn,
        verbose=True,
        tokenizer=lambda x: tokenizer(x).input_ids,
        trigram=trigram
    )

    train_vector_map = {
        "falcon-logprobs": lambda file: train_falcon_logprobs[file],
        "trigram-logprobs": lambda file: train_trigram_logprobs[file],
        "unigram-logprobs": lambda file: train_unigram_logprobs[file],
    }
    test_vector_map = {
        "falcon-logprobs": lambda file: test_falcon_logprobs[file],
        "trigram-logprobs": lambda file: test_trigram_logprobs[file],
        "unigram-logprobs": lambda file: test_unigram_logprobs[file],
    }

    def get_exp_featurize(best_features, vector_map):
        def calc_features(file, exp):
            exp_tokens = get_words(exp)
            curr = vector_map[exp_tokens[0]](file)

            for i in range(1, len(exp_tokens)):
                if exp_tokens[i] in vec_functions:
                    next_vec = vector_map[exp_tokens[i + 1]](file)
                    curr = vec_functions[exp_tokens[i]](curr, next_vec)
                elif exp_tokens[i] in scalar_functions:
                    return scalar_functions[exp_tokens[i]](curr)

        def exp_featurize(file):
            return np.array([calc_features(file, exp) for exp in best_features])

        return exp_featurize

    print("Grabbing features and applying normalization")
    train_data = train_generate_dataset_fn(get_exp_featurize(best_features, train_vector_map), verbose=True, split=train)
    train_data, mu, sigma = normalize(train_data, ret_mu_sigma=True)
    test_data = test_generate_dataset_fn(get_exp_featurize(best_features, test_vector_map), verbose=True)
    test_data = normalize(test_data, mu, sigma)
    print(f"{mu=}")
    print(f"{sigma=}")

    def train_lr(train_data, train_indices, test_data, test_indices):
        # train model
        model = LogisticRegression(max_iter=1000)
        model.fit(train_data, train_labels) # no need indices for splits

        # fit on test
        preds = model.predict_proba(test_data[test]) # (n_sample, 2)
        preds = preds[:, 1]
        metric_score = roc_auc_score(y_true=test_labels[test], y_score=preds)

        return preds, metric_score, model
    
    # user all train and all test
    print("Fitting LR")
    test_preds, auroc_score, model = train_lr(train_data, train, test_data, test)
    print(f"{args.domain} AUROC = {auroc_score}")

    # save model and mu/sigma for normalization
    with open(os.path.join(results_folder, f"model_{args.domain}{'_sample' if args.do_sample else ''}.pkl"), "wb") as f:
        pickle.dump({"model": model, "mu": mu, "sigma": sigma}, f)

    # save test set predictions
    pred_folder = "predictions_m4gt/"
    pred_file = args.domain + ("_sample" if args.do_sample else "") + ".preds"
    os.makedirs(pred_folder, exist_ok=True)
    pd.DataFrame({
        "id": test_ids,
        "score": test_preds
    }).to_csv(os.path.join(pred_folder, pred_file), sep="\t", index=False)
