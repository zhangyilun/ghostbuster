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
import numpy as np
import dill as pickle

from nltk.util import ngrams
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


results_folder = "results_raid_500k"


parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, required=True)
parser.add_argument("--feature_select", action="store_true")
parser.add_argument("--classify", action="store_true")
parser.add_argument("--do_sample", action="store_true")
args = parser.parse_args()

domains = [ args.domain ]

train_datasets = [
    Dataset("normal", f"data/raid_500k_{d}/train/{t}")
    for d in domains for t in ["human", "gpt"]
]
test_datasets = [
    Dataset("normal", f"data/raid_500k_{d}/test/{t}")
    for d in domains for t in ["human", "gpt"]
]

'''
# output of --feature_select
best_features = [
    "trigram-logprobs v-add unigram-logprobs v-> falcon-logprobs s-var",
    "trigram-logprobs v-div unigram-logprobs v-div trigram-logprobs s-avg-top-25",
    "unigram-logprobs v-mul falcon-logprobs s-avg",
    "trigram-logprobs v-mul unigram-logprobs v-div trigram-logprobs s-avg",
    "trigram-logprobs v-< unigram-logprobs v-mul falcon-logprobs s-avg-top-25",
    "trigram-logprobs v-mul unigram-logprobs v-sub falcon-logprobs s-min",
    "trigram-logprobs v-mul unigram-logprobs s-avg",
    "trigram-logprobs v-< unigram-logprobs v-sub falcon-logprobs s-avg",
    "trigram-logprobs v-> unigram-logprobs v-add falcon-logprobs s-avg",
    "trigram-logprobs v-div llama-logprobs v-div trigram-logprobs s-min",
]
'''

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
    num_tokens=511, # max_length-1
):
    falcon_logprobs = {}
    trigram_logprobs, unigram_logprobs = {}, {}

    if verbose:
        print("Loading logprobs into memory")

    file_names = generate_dataset(lambda file: file, verbose=False)
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
# train size: 105443
# test size: 500k
train = np.arange(train_labels.size)
test = np.arange(test_labels.size)

# sample 6000
sample_size = 6000
if args.do_sample:
    np.random.seed(0)
    train = np.random.choice(train, size=sample_size, replace=False)
print(f"{len(train)=}, {len(test)=}")


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
        train_unigram_logprobs,
    ) = get_all_logprobs(
        train_generate_dataset_fn,
        verbose=True,
        tokenizer=lambda x: tokenizer(x)["input_ids"],
        trigram=trigram,
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
            lambda file: calc_features(file, exp)
        ).reshape(-1, 1)

    best_features = select_features(exp_to_data, train_labels, verbose=True, to_normalize=True, indices=train)
    print(best_features)
    
    # save best features
    outfolder_name = "best_features_{args.domain}"
    if args.do_sample:
        outfolder_name += "_sample"
    outfolder_name += ".txt"
    with open(os.path.join(results_folder, outfolder_name), "w") as f:
        for feat in best_features:
            f.write(feat + "\n")


if args.classify:
    (
        falcon_logprobs,
        trigram_logprobs,
        unigram_logprobs,
    ) = get_all_logprobs(
        train_generate_dataset_fn,
        verbose=True,
        tokenizer=lambda x: tokenizer(x)["input_ids"],
        trigram=trigram,
    )

    vector_map = {
        "falcon-logprobs": lambda file: falcon_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file],
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

    train_data = train_generate_dataset_fn(get_exp_featurize(best_features, vector_map))
    train_data = normalize(train_data)
    test_data = test_generate_dataset_fn(get_exp_featurize(best_features, vector_map))
    test_data = normalize(test_data)

    def train_lr(data, train, test):
        model = LogisticRegression()
        model.fit(data[train], labels[train])
        # return f1_score(labels[test], model.predict(data[test]))
        return roc_auc_score(labels[test], model.predict(data[test]))

    print(
        f"In-Domain AUC: {train_lr(data, indices_dict['gpt_train'] + indices_dict['human_train'], indices_dict['gpt_test'] + indices_dict['human_test'])}"
    )

    for test_domain in domains:
        train_indices = []
        for train_domain in domains:
            if train_domain == test_domain:
                continue

            train_indices += (
                indices_dict[f"gpt_{train_domain}_train"]
                + indices_dict[f"human_{train_domain}_train"]
            )

        print(
            f"Out-Domain AUC ({test_domain}): {train_lr(data, train_indices, indices_dict[f'gpt_{test_domain}_test'] + indices_dict[f'human_{test_domain}_test'])}"
        )
