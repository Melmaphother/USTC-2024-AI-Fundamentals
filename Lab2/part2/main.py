import torch
import torch.nn as nn
import torch.optim as optim
from args import args
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from datautils import ShakespeareDataset, Tokenizer
from TransformerDecoderLayer import SparseMoETransformer
from train import Trainer, test
from transformers import BertTokenizer
import tiktoken


def main():
    # 加载分词器
    if args.tokenizer_mode == 'custom':
        tokenizer = Tokenizer(args.datapath)
    elif args.tokenizer_mode == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.tokenizer_mode == 'tiktoken':
        tokenizer = tiktoken.get_encoding("cl100k_base")
    else:
        raise ValueError("Invalid tokenizer mode. Choose from 'custom', 'bert', 'tiktoken'.")

    # 加载数据集
    dataset = ShakespeareDataset(args.datapath, args.tokenizer_mode, tokenizer, args.chunk_size)

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
    ).to(args.device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练和验证
    trainer = Trainer(args, model, train_dataloader, val_dataloader, criterion, optimizer)
    trainer.train()

    # 加载最佳模型并测试
    model.load_state_dict(torch.load("models/best_model.pth"))
    test(args, model, test_dataloader, criterion)


def generate_text(input_text: str, max_len: int = 100):
    if args.tokenizer_mode == 'custom':
        tokenizer = Tokenizer(args.datapath)
        encoded_text = tokenizer.encode(input_text).unsqueeze(0).to(args.device)
    elif args.tokenizer_mode == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded_text = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt').to(args.device)
    elif args.tokenizer_mode == 'tiktoken':
        tokenizer = tiktoken.get_encoding("cl100k_base")
        encoded_text = torch.tensor(tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(args.device)
    else:
        raise ValueError("Invalid tokenizer mode. Choose from 'custom', 'bert', 'tiktoken'.")

    # 加载数据集
    dataset = ShakespeareDataset(args.datapath, args.tokenizer_mode, tokenizer, args.chunk_size)

    # 设置词汇表大小
    args.vocab_size = dataset.get_vocab_size()

    # 加载模型
    model = SparseMoETransformer(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_experts=args.num_experts,
        active_experts=args.active_experts,
        dropout=0.1
    ).to(args.device)

    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()

    if args.tokenizer_mode == 'custom':
        gen_tokens = model.generate(encoded_text, max_len)[0].tolist()
        gen_text = tokenizer.decode(gen_tokens)
    elif args.tokenizer_mode == 'bert':
        gen_tokens = model.generate(encoded_text, max_len)[0].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    elif args.tokenizer_mode == 'tiktoken':
        gen_tokens = model.generate(encoded_text, max_len)[0].tolist()
        gen_text = tokenizer.decode(gen_tokens)
    else:
        raise ValueError("Invalid tokenizer mode. Choose from 'custom', 'bert', 'tiktoken'.")

    print(gen_text)


if __name__ == '__main__':
    # main()
    generate_text("To be or not to be, that is the question:", max_len=100)
    generate_text("I could pick my lance", max_len=100)
