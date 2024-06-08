import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import tiktoken


class Tokenizer:
    def __init__(self, datapath: str):
        with open(datapath, 'r', encoding='utf-8') as f:
            self.dataset = f.read()
        self.__gen_vocab()

    def __gen_vocab(self):
        # 开始符号 <CLS>, 分隔符号 <SEP>, 未知符号 <UNK>
        self.char2idx = {'<UNK>': 0, '<CLS>': 1, '<SEP>': 2}
        self.idx2char = {0: '<UNK>', 1: '<CLS>', 2: '<SEP>'}

        for idx, char in enumerate(set(self.dataset), start=1):
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode(self, sentence: str) -> torch.Tensor:
        indices = [self.char2idx.get(char, 0) for char in sentence]
        return torch.tensor([1] + indices + [2], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        chars = [self.idx2char[_id.item()] for _id in ids]
        return ''.join(chars[1:-1])


class ShakespeareDataset(Dataset):
    def __init__(self, datapath: str, tokenizer: str = 'custom', chunk_size: int = 100):
        with open(datapath, 'r', encoding='utf-8') as f:
            self.dataset = f.read()
        self.chunk_size = chunk_size
        if tokenizer == 'custom':
            self.tokenizer = Tokenizer(datapath)
        elif tokenizer == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif tokenizer == 'tiktoken':
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.encoded_dataset = self.tokenizer.encode(self.dataset)

    def __len__(self):
        return len(self.encoded_dataset) - self.chunk_size

    def __getitem__(self, idx):
        chunk = self.encoded_dataset[idx:idx + self.chunk_size]
        label = self.encoded_dataset[idx + 1:idx + self.chunk_size + 1]
        return chunk, label
