import torch
import torch.nn as nn
import torch.optim as optim
from args import args
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from datautils import ShakespeareDataset
from TransformerDecoderLayer import SparseMoETransformer
from train import Trainer, test


def main():
    # 加载数据集
    dataset = ShakespeareDataset(args.datapath, args.tokenizer, args.chunk_size)

    # 设置词汇表大小
    args.vocab_size = dataset.get_vocab_size()

    # 划分数据集，并创建 DataLoader
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = SparseMoETransformer(
        vocab_size=args.vocab_size,
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
    trainer = Trainer(args, model, train_dataloader, val_dataloader, criterion, optimizer)
    trainer.train()

    # 加载最佳模型并测试
    model.load_state_dict(torch.load("models/best_model.pth"))
    test(args, model, test_dataloader, criterion)


if __name__ == '__main__':
    main()
