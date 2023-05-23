import torch
from torch import nn
from torch_geometric.loader import DataLoader
import wandb
from dataclasses import dataclass
from tqdm import tqdm
import argparse

from dataset import DrugCombDataset
from models import CongFuBasedModel
from utils import get_datasets, get_mol_dict, calculate_auprc, calculate_roc_auc

WANDB_PROJECT = "your_wandb_project_name"
WANDB_ENTITY = "your_wandb_entity"


@dataclass
class TrainConfiguration:
    synergy_score: str
    transductive: bool
    inductive_set_name: str
    fold_number: int
    batch_size: int
    lr: float
    number_of_epochs: int
    data_folder_path: str
    

def evaluate_mlp(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> None:
    model.eval()

    epoch_preds, epoch_labels = [], []
    epoch_loss = 0.0

    for batch in loader:
        batch = [tensor.to(device) for tensor in batch]
        drugA, drugB, cell_line, target = batch

        with torch.no_grad():
            output = model(drugA, drugB, cell_line)

        loss = loss_fn(output, target)
        epoch_preds.append(output.detach().cpu())
        epoch_labels.append(target.detach().cpu())
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)

    auprc = calculate_auprc(epoch_labels, epoch_preds)
    auc = calculate_roc_auc(epoch_labels, epoch_preds)
    
    if wandb.run is not None:
        wandb.log({"val_auprc": auprc, "val_auc": auc, "val_loss": epoch_loss})

def train_model(model: nn.Module, config: TrainConfiguration, device: torch.device) -> None:
    dataset, train_dataset, test_dataset, cell_lines = get_datasets(config.data_folder_path, config.fold_number, config.synergy_score, config.transductive, config.inductive_set_name)
    mol_mapping = get_mol_dict(dataset)

    train_set = DrugCombDataset(train_dataset, cell_lines, mol_mapping)
    test_set = DrugCombDataset(test_dataset, cell_lines, mol_mapping)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=2, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.number_of_epochs)

    loss_fn = nn.BCEWithLogitsLoss()

    model.train()

    for _ in tqdm(range(config.number_of_epochs)):
        epoch_preds, epoch_labels = [], []
        epoch_loss = 0.0

        for batch in train_loader:
            batch = [tensor.to(device) for tensor in batch]
            drugA, drugB, cell_line, target = batch

            optimizer.zero_grad()
            output = model(drugA, drugB, cell_line)
            loss = loss_fn(output, target)

            epoch_preds.append(output.detach().cpu())
            epoch_labels.append(target.detach().cpu())
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_loader)
        epoch_preds = torch.cat(epoch_preds)
        epoch_labels = torch.cat(epoch_labels)

        auprc = calculate_auprc(epoch_labels, epoch_preds)
        auc = calculate_roc_auc(epoch_labels, epoch_preds)

        if wandb.run is not None:
            wandb.log({"train_auprc": auprc, "train_auc": auc, "train_loss": epoch_loss})

        evaluate_mlp(model, test_loader, loss_fn, device)
    

def train(config):
    if config.with_wandb:
        wandb.init(config=config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
        print(f'Hyper parameters:\n {wandb.config}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CongFuBasedModel(
        num_layers=config.num_layers,
        inject_layer = config.inject_layer,
        emb_dim = config.emb_dim,
        feature_dim = config.feature_dim,
        context_dim = config.context_dim,
        device=device
    )

    train_configuraion = TrainConfiguration(
        synergy_score=config.synergy_score,
        transductive = config.transductive,
        inductive_set_name = config.inductive_set_name,
        lr = config.lr,
        number_of_epochs = config.number_of_epochs,
        data_folder_path=config.data_folder_path,
        fold_number = config.fold_number,
        batch_size=config.batch_size
    )

    train_model(model, train_configuraion, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CongFu-based model')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--inject_layer', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--context_dim', type=int, default=908)
    parser.add_argument('--synergy_score', type=str, default="loewe")
    parser.add_argument('--transductive', action='store_true')
    parser.add_argument('--inductive_set_name', type=str, default="leave_comb")
    parser.add_argument('--fold_number', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--number_of_epochs', type=int, default=100)
    parser.add_argument('--data_folder_path', type=str, default="data/preprocessed/")
    parser.add_argument('--with_wandb', action='store_true')

    config = parser.parse_args()
    train(config)