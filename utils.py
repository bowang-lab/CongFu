import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from typing import Tuple
import json
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def split_fold(dataset, fold: dict[str, list[int]]):
    train_indices, test_indices = fold["train"], fold["test"]
    X_train = dataset.iloc[train_indices]
    X_test = dataset.iloc[test_indices]

    return X_train, X_test

def calculate_roc_auc(targets, preds):
    return roc_auc_score(targets, preds)


def calculate_auprc(targets, preds):
    precision_scores, recall_scores, __ = precision_recall_curve(targets, preds)

    return auc(recall_scores, precision_scores)

def get_datasets(data_folder_path: str, fold_number: int, synergy_score: str, transductive: bool, inductive_set_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_lines = pd.read_feather(data_folder_path + f"drugcomb/cell_lines.feather").set_index("cell_line_name")
    cell_lines = cell_lines.astype(np.float32)

    if transductive:
        dataset = pd.read_feather(data_folder_path + f"drugcomb/{synergy_score}/transductive/{synergy_score}.feather")

        with open(data_folder_path + f"drugcomb/{synergy_score}/transductive/{synergy_score}.json") as f:
            folds = json.load(f)

        fold = folds[f"fold_{fold_number}"]
        train_dataset, test_dataset = split_fold(dataset, fold)
    else:
        inductive_set_name = inductive_set_name

        train_dataset = pd.read_feather(data_folder_path + f"drugcomb/{synergy_score}/{inductive_set_name}/train_{fold_number}.feather")
        test_dataset = pd.read_feather(data_folder_path + f"drugcomb/{synergy_score}/{inductive_set_name}/test_{fold_number}.feather")
        dataset = pd.concat((train_dataset, test_dataset))

    return dataset, train_dataset, test_dataset, cell_lines

def _get_drug_tokens(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def get_mol_dict(df):
    mols = pd.concat([
        df.rename(columns={'Drug1_ID': 'id', 'Drug1': 'drug'})[['id', 'drug']],
        df.rename(columns={'Drug2_ID': 'id', 'Drug2': 'drug'})[['id', 'drug']]
    ],
        axis=0, ignore_index=True
    ).drop_duplicates(subset=['id'])

    dct = {}
    for _, x in tqdm(mols.iterrows(), total=len(mols)):
        dct[x['id']] = _get_drug_tokens(x['drug'])
    return dct