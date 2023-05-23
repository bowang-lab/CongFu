from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import os

def add_cosmic_ids(drugcomb: pd.DataFrame,
                   read_path: str) -> pd.DataFrame:
    cosmic_ids = pd.read_csv(read_path + 'cellosaurus_cosmic_ids.txt', sep=',', header=None)
    
    cosmic_ids = cosmic_ids.dropna()
    
    mapping_cosmic = dict(zip(cosmic_ids[0], cosmic_ids[1]))
    drugcomb["cosmicId"] = drugcomb['cell_line_name'].map(mapping_cosmic)
    
    return drugcomb

def get_smiles(drugcomb: pd.DataFrame,
               read_path: str
              ) -> pd.DataFrame:

    drugs = pd.read_json(read_path + 'drugs.json')
    
    drugs = drugs[drugs['smiles']!='NULL']
    drugs = drugs[~drugs['smiles'].str.contains("Antibody")]
    
    drugs['smiles'] = drugs['smiles'].apply(lambda x: x.split('; ')[-1])
    drugs['smiles'] = drugs['smiles'].apply(lambda x: x.split(';')[-1])
    
    mapping_smiles = dict(zip(drugs['dname'], drugs['smiles']))

    drugcomb["drug_row_smiles"] = drugcomb['drug_row'].map(mapping_smiles)
    drugcomb["drug_col_smiles"] = drugcomb['drug_col'].map(mapping_smiles)
    drugcomb = drugcomb.dropna(subset=['drug_row_smiles', 'drug_col_smiles'])
    
    return drugcomb

def get_target(drugcomb: pd.DataFrame, synergy_score: str,
               synergistic_threshold: float, antagonistic_threshold: float) -> pd.DataFrame:
    
    def synergy_threshold(value):
        if (value >= synergistic_threshold):
            return 1
        if (value <= antagonistic_threshold):
            return 0
        return np.nan
    
    drugcomb['target'] = drugcomb[f'synergy_{synergy_score}'].apply(synergy_threshold)
    drugcomb = drugcomb.dropna(subset=['target'])
    
    return drugcomb

def formatting(drugcomb: pd.DataFrame) -> pd.DataFrame:
    drugcomb = drugcomb[drugcomb['synergy_loewe']!='\\N']
    drugcomb = drugcomb.dropna(subset=['drug_col','cosmicId'])
    drugcomb['synergy_loewe'] = drugcomb['synergy_loewe'].astype(float)
    drugcomb['cosmicId'] = drugcomb['cosmicId'].astype(int)
    
    return drugcomb

def rename_and_crop(drugcomb: pd.DataFrame) -> pd.DataFrame:
    
    drugcomb = drugcomb.rename(
        columns={"drug_row": "Drug1_ID",
                 "drug_col": "Drug2_ID",
                 "cell_line_name": "Cell_Line_ID",
                 "drug_row_smiles": "Drug1",
                 "drug_col_smiles": "Drug2"})
    
    cols_to_keep = ["Drug1_ID", "Drug2_ID", "Cell_Line_ID", "Drug1", "Drug2", "target"]
    drugcomb = drugcomb[cols_to_keep].reset_index(drop=True)
    
    return drugcomb

def drop_duplicates(drugcomb:pd.DataFrame) -> pd.DataFrame:
    
    dup_to_drop = []

    drugcomb_copy = drugcomb.copy()
    cols = ['drug_row', 'drug_col', "cell_line_name"]
    drugcomb_copy[cols] = np.sort(drugcomb_copy[cols].values, axis=1)
    dup = drugcomb_copy.duplicated(subset=cols, keep=False)

    dup_score = drugcomb_copy[dup][cols+['target']]
    dup_val = dup_score.duplicated(keep=False)
    dup_val_true = drugcomb_copy[dup][cols+['target']][dup_val]  # same triplets and class
    dup_val_false = drugcomb_copy[dup][cols+['target']][~dup_val]  # same triplets, another class

    dup_to_drop += list(dup_val_true[dup_val_true.duplicated(keep="first")].index)
    
    dup2 = pd.concat([dup_val_false, dup_val_true[~dup_val_true.duplicated(keep="first")]], axis=0)
    dup2_val = dup2.duplicated(subset=(cols), keep=False)
    dup_to_drop += list(dup2[dup2_val].sort_values(cols).index)

    return drugcomb.drop(index=dup_to_drop)

def process_cell_lines(drugcomb: pd.DataFrame,
                      read_path: str
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    rma = pd.read_csv(read_path + 'Cell_line_RMA_proc_basalExp.txt', sep='\t')
    
    cosmic_found = set(drugcomb.cosmicId)
    cosmic_intersect = list(set(["DATA."+str(c) for c in cosmic_found]).intersection(set(rma.columns)))
    
    drugcomb = drugcomb[drugcomb.cosmicId.isin([int(c[len("DATA."):]) for c in cosmic_intersect])]
    
    landmark_genes = pd.read_csv(read_path + 'L1000genes.txt', sep='\t')
    landmark_genes = list(landmark_genes.loc[landmark_genes['Type']=="landmark",'Symbol'])
    
    rma_landm = rma[rma['GENE_SYMBOLS'].isin(landmark_genes)]
    rma_landm = rma_landm.drop(['GENE_SYMBOLS','GENE_title'], axis=1).T
    
    cell_line_mapping = dict(zip('DATA.'+drugcomb['cosmicId'].astype(str),drugcomb['cell_line_name']))
    rma_landm.index = rma_landm.index.map(cell_line_mapping)
    rma_landm = rma_landm[~rma_landm.index.isna()]
    
    scaler = StandardScaler()
    scaler.fit(rma_landm)

    cell_line_feats = pd.DataFrame(
        scaler.transform(rma_landm),
        columns=[f'feat_{i}' for i in range(rma_landm.shape[1])],
        # index=rma_landm.index
    )
    cell_line_feats["cell_line_name"] = rma_landm.index
    return drugcomb, cell_line_feats

def preprocess_drugcomb(synergy_score: str,
                        synergistic_thresh: float,
                        antagonistic_thresh: float,
                        save_cell_lines: bool,
                        read_path: str,
                        save_path: str,
                       ) -> None:

    drugcomb = pd.read_csv(read_path + 'summary_v_1_5.csv')
    
    
    drugcomb = add_cosmic_ids(drugcomb, read_path)
    drugcomb = formatting(drugcomb)
    drugcomb = get_smiles(drugcomb, read_path)
    drugcomb = get_target(drugcomb, synergy_score, synergistic_thresh, antagonistic_thresh)
    drugcomb = drop_duplicates(drugcomb)
    drugcomb, cell_line_feats = process_cell_lines(drugcomb, read_path)
    drugcomb = rename_and_crop(drugcomb)

    drugcomb = drugcomb.reset_index(drop=True)

    directory_name = save_path + synergy_score
    os.makedirs(directory_name, exist_ok=True)
        
    drugcomb.to_feather(f"{directory_name}/{synergy_score}.feather")
    
    if save_cell_lines:
        cell_line_feats = cell_line_feats.reset_index(drop=True)
        cell_line_feats.to_feather(save_path + f"cell_lines.feather")

if __name__ == "__main__":
    print('Preprocessing DrugComb')

    parser = argparse.ArgumentParser(description='Preprocessing Drug Comb')
    parser.add_argument('--synergy_score', type=str, default="loewe")
    parser.add_argument('--synergistic_thresh', type=float, default=10.0)
    parser.add_argument('--antagonistic_thresh', type=float, default=-10.0)
    parser.add_argument('--save_cell_lines', type=bool, default=True)
    parser.add_argument('--read_path', type=str, default="data/init/")
    parser.add_argument('--save_path', type=str, default="data/preprocessed/")
    args = parser.parse_args()
    print(args)

    preprocess_drugcomb(
        args.synergy_score,
        args.synergistic_thresh,
        args.antagonistic_thresh,
        args.save_cell_lines,
        args.read_path,
        args.save_path
    )

    print("Done")
