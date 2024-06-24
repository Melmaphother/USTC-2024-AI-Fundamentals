import torch
from tqdm import tqdm
from datautils import generate_tgt_mask
from torch.utils.tensorboard import SummaryWriter


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
        self.save_path = args.save_path
        self.model_path = args.model_path
        self.writer = SummaryWriter(log_dir=self.save_path)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            train_loss = self._train_single_epoch()
            val_loss = self._val_single_epoch()

            print(f"Epoch {epoch + 1}/{self.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.writer.add_scalar("Train Loss", train_loss, epoch)
            self.writer.add_scalar("Val Loss", val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.model_path}/best_model.pth")

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
    writer = SummaryWriter(log_dir=args.save_path)

    model.eval()
    test_loss = 0
    for batch in tqdm(test_dataloader, desc="Testing"):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        tgt_mask = generate_tgt_mask(inputs.size(1)).to(device)

        output = model(inputs, tgt_mask=tgt_mask)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

        test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")
    writer.add_scalar("Test Loss", test_loss)
