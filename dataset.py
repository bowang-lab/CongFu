import torch
from torch.utils.data import Dataset

DRUG1_ID_COLUMN_NAME = "Drug1_ID"
DRUG2_ID_COLUMN_NAME= "Drug2_ID"
CELL_LINE_COLUMN_NAME = "Cell_Line_ID"

class DrugCombDataset(Dataset):
    def __init__(self, drugcomb, cell_lines, mol_mapping, transform=None):
        self.drugcomb = drugcomb
        self.mol_mapping = mol_mapping
        self.cell_lines = cell_lines
        self.targets = torch.from_numpy(drugcomb['target'].values)
        self.transform = transform

    def __len__(self):
        return len(self.drugcomb)

    def __getitem__(self, idx):
        sample = self.drugcomb.iloc[idx]

        drug1 = sample[DRUG1_ID_COLUMN_NAME]
        drug2 = sample[DRUG2_ID_COLUMN_NAME]
        drug1_tokens = self.mol_mapping[drug1]
        drug2_tokens = self.mol_mapping[drug2]

        if self.transform:
            drug1_tokens = self.transform(drug1_tokens)
            drug2_tokens = self.transform(drug2_tokens)

        cell_line_name = sample[CELL_LINE_COLUMN_NAME]
        cell_line_embeddings = self.cell_lines.loc[cell_line_name].values.flatten()
        cell_line_embeddings = torch.tensor(cell_line_embeddings)

        target = self.targets[idx].unsqueeze(-1).float()
        
        return (drug1_tokens, drug2_tokens, cell_line_embeddings, target)