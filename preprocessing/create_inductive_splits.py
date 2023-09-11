import pandas as pd
import random
import numpy as np
import os
import argparse

def sort_drugs(x1: str, x2: str) -> str:
    if x1 > x2:
        return x1 + '_' + x2
    else:
        return x2 + '_' + x1
    
def create_inductive_splits(synergy_score: str,
                            num_splits: int,
                            read_path: str,
                            save_path: str
                           ) -> None:
        
    df = pd.read_feather(read_path + f'{synergy_score}/{synergy_score}.feather')
    df['sorted_drugs'] = df.apply(lambda x: sort_drugs(x['Drug1_ID'], x['Drug2_ID']), axis=1)
    
    unique_combs = list(set(df['sorted_drugs']))
    unique_drugs = list(set(df['Drug1_ID']).union(set(df['Drug2_ID'])))
    
    random.shuffle(unique_combs)
    random.shuffle(unique_drugs)
    
    splitted_combs = np.array_split(unique_combs, num_splits)
    splitted_drugs = np.array_split(unique_drugs, num_splits)
    
    for split_type in ['leave_comb', 'leave_drug']:
        
        local_save_path = os.path.join(save_path, synergy_score, split_type)
        os.makedirs(local_save_path, exist_ok=True)
        
        for i in range(num_splits):
            
            if split_type == 'leave_comb':
                test_mask = df['sorted_drugs'].isin(splitted_combs[i])
                
            if split_type == 'leave_drug':
                test_mask = (df['Drug1_ID'].isin(splitted_drugs[i])|df['Drug2_ID'].isin(splitted_drugs[i]))

            test = df[test_mask].reset_index(drop=True)
            train = df[~test_mask].reset_index(drop=True)

            train.to_feather(os.path.join(local_save_path, f'train_{i}.feather'))
            test.to_feather(os.path.join(local_save_path, f'test_{i}.feather'))
    
    
if __name__ == "__main__":
    print('Inductive Split DrugComb')

    parser = argparse.ArgumentParser(description='Inductive Split DrugComb')
    parser.add_argument('--synergy_score', type=str, default="loewe")
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--read_path', type=str, default="data/preprocessed/")
    parser.add_argument('--save_path', type=str, default="data/preprocessed/")
    args = parser.parse_args()
    print(args)

    create_inductive_splits(
        args.synergy_score,
        args.num_splits,
        args.read_path,
        args.save_path,
    )

    print("Done")
