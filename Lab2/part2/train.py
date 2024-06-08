import torch
from tqdm import tqdm


def generate_tgt_mask(seq_len):
    """生成上三角的掩蔽矩阵，防止看到未来的词"""
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    return mask


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # tgt_mask
        tgt_mask = generate_tgt_mask(inputs.size(1)).to(device)

        optimizer.zero_grad()
        output = model(inputs, tgt_mask=tgt_mask)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # tgt_mask
            tgt_mask = generate_tgt_mask(inputs.size(1)).to(device)

            output = model(inputs, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
