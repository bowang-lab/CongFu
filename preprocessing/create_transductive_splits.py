from sklearn.model_selection import StratifiedKFold
import pandas as pd
import json
import argparse
import os

RANDOM_SEED = 42
TARGET_COLUMN_NAME = "target"
SYNERGY_SCORES = ["loewe", "bliss", "hsa", "zip"]

def split_k_fold(config: argparse.Namespace):
    folds = {}
    read_path = os.path.join(config.data_folder_path, config.synergy_score, f"{config.synergy_score}.feather")
    drugcomb = pd.read_feather(read_path)
    drugcomb_indexes = drugcomb.index.values
    drugcomb_targets = drugcomb[TARGET_COLUMN_NAME].values

    skfold = StratifiedKFold(n_splits=config.fold_number, shuffle=True, random_state=RANDOM_SEED)

    for index, (train_indexes, test_indexes) in enumerate(skfold.split(drugcomb_indexes, drugcomb_targets)):
        folds[f"fold_{index}"] = {"train": train_indexes.tolist(),
                                  "test": test_indexes.tolist()}

    save_path = os.path.join(config.data_folder_path, config.synergy_score, f"{config.synergy_score}.json")
    with open(save_path, "w") as file:
        json.dump(folds, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split K Folds')
    parser.add_argument('--fold_number', type=int, default=5)
    parser.add_argument('--synergy_score', type=str, default="loewe")
    parser.add_argument('--data_folder_path', type=str, default="data/preprocessed/")

    config = parser.parse_args()
    
    split_k_fold(config)