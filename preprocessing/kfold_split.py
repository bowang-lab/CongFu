from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import argparse

RANDOM_SEED = 42
TARGET_COLUMN_NAME = "target"
SYNERGY_SCORES = ["loewe", "bliss", "hsa", "zip"]

def split_k_fold(synergy_score: str, config: argparse.Namespace):
    folds = {}

    drugcomb = pd.read_feather(config.data_folder_path + f"{synergy_score}/transductive/{synergy_score}.feather")
    drugcomb_indexes = drugcomb.index.values
    drugcomb_targets = drugcomb[TARGET_COLUMN_NAME].values

    skfold = StratifiedKFold(n_splits=config.fold_number, shuffle=True, random_state=RANDOM_SEED)

    for index, (train_indexes, test_indexes) in enumerate(skfold.split(drugcomb_indexes, drugcomb_targets)):
        folds[f"fold_{index}"] = {"train": train_indexes.tolist(),
                                  "test": test_indexes.tolist()}

    with open(config.data_folder_path + f"{synergy_score}/transductive/{synergy_score}.json", "w") as file:
        json.dump(folds, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split K Folds')
    parser.add_argument('--fold_number', type=int, default=5)
    parser.add_argument('--data_folder_path', type=str, default="data/preprocessed/drugcomb")

    config = parser.parse_args()
    
    for synergy_score in SYNERGY_SCORES:
        split_k_fold(synergy_score, config)