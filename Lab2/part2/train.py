import torch
from tqdm import tqdm
import os


def generate_tgt_mask(seq_len):
    """生成上三角的掩蔽矩阵，防止看到未来的词"""
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    return mask


class Trainer:
    def __init__(self, args, model, train_dataloader, val_dataloader, criterion, optimizer):
        self.args = args
        self.device = args.device
        self.epochs = args.epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.model.to(self.device)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            train_loss = self._train_single_epoch()
            val_loss = self._val_single_epoch()

            print(f"Epoch {epoch + 1}/{self.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists("models"):
                    os.makedirs("models")
                torch.save(self.model.state_dict(), "models/best_model.pth")

    def _train_single_epoch(self):
        self.model.train()
        epoch_loss = 0
        for batch in tqdm(self.train_dataloader, desc="Training"):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # tgt_mask
            tgt_mask = generate_tgt_mask(inputs.size(1)).to(self.device)

            self.optimizer.zero_grad()
            output = self.model(inputs, tgt_mask=tgt_mask)
            loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_dataloader)

    @torch.no_grad()
    def _val_single_epoch(self):
        self.model.eval()
        epoch_loss = 0
        for batch in tqdm(self.val_dataloader, desc="Validation"):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # tgt_mask
            tgt_mask = generate_tgt_mask(inputs.size(1)).to(self.device)

            output = self.model(inputs, tgt_mask=tgt_mask)
            loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))

            epoch_loss += loss.item()

        return epoch_loss / len(self.val_dataloader)


@torch.no_grad()
def test(args, model, test_dataloader, criterion):
    device = args.device
    model.to(device)

    model.eval()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in tqdm(test_dataloader, desc="Testing"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            tgt_mask = generate_tgt_mask(inputs.size(1)).to(device)

            output = model(inputs, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs}: Test Loss: {epoch_loss / len(test_dataloader):.4f}")
