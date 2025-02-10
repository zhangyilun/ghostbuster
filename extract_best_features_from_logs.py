import os
import argparse

import re


parser = argparse.ArgumentParser()
parser.add_argument("--log_file", type=str, required=True)
parser.add_argument("--domain", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
args = parser.parse_args()


with open(args.log_file) as f:
    lines = f.read().splitlines()

# get lines with "Best Feature"
feat_lines = [l for l in lines if "Best Feature" in l]

# get feature names
best_features = []
pat = r"Best Feature: (.*?), New Score:"
for l in feat_lines:
    match = re.search(pat, l)
    if match:
        best_features.append(match.group(1))

print(best_features)

# save
with open(os.path.join(args.out_dir, f"best_features_{args.domain}.txt"), "w") as f:
    for l in best_features:
        f.write(l + "\n")
