import torch.nn as nn
import torch.optim as optim
from args import args
from datautils import Tokenizer, create_dataloader
from TransformerDecoderLayer import SparseMoETransformer
from train import train, validate


def main():
    tokenizer = Tokenizer(args.datapath)
    train_dataloader, val_dataloader = create_dataloader(args.datapath, 'custom', args.chunk_size, args.batch_size)

    # 初始化模型
    model = SparseMoETransformer(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_experts=args.num_experts,
        active_experts=args.active_experts,
        dropout=0.1
    )

    model.to(args.device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练和验证
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, args.device)
        valid_loss = validate(model, val_dataloader, criterion, args.device)

        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')


if __name__ == '__main__':
    main()
