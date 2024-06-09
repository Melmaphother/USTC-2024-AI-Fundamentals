import torch
from torch.utils.data import Dataset
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"
os.environ['ALL_PROXY'] = "socks5://127.0.0.1:7897"


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

    def encode(self, sentence):
        indices = [self.char2idx.get(char, 0) for char in sentence]
        return torch.tensor([1] + indices + [2], dtype=torch.long)

    def decode(self, ids):
        chars = [self.idx2char[_id.item()] for _id in ids]
        return ''.join(chars[1:-1])

    def get_vocab_size(self):
        return len(self.char2idx)


class ShakespeareDataset(Dataset):
    def __init__(self, datapath: str, tokenizer_mode: str, tokenizer, chunk_size: int):
        with open(datapath, 'r', encoding='utf-8') as f:
            self.dataset = f.read()
        self.chunk_size = chunk_size
        self.tokenizer_mode = tokenizer_mode
        self.tokenizer = tokenizer
        if tokenizer_mode == 'custom':
            self.vocab_size = tokenizer.get_vocab_size()
            self.encoded_dataset = tokenizer.encode(self.dataset)
        elif tokenizer_mode == 'bert':
            self.vocab_size = tokenizer.vocab_size
            self.encoded_dataset = tokenizer.encode(self.dataset, add_special_tokens=True)
        elif tokenizer_mode == 'tiktoken':
            self.vocab_size = None  # TODO
            self.encoded_dataset = tokenizer.encode(self.dataset)

    def __len__(self):
        return len(self.encoded_dataset) - self.chunk_size

    def __getitem__(self, idx):
        chunk = self.encoded_dataset[idx:idx + self.chunk_size]
        label = self.encoded_dataset[idx + 1:idx + self.chunk_size + 1]
        # 转成 tensor
        chunk = torch.tensor(chunk, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return chunk, label

    def get_vocab_size(self):
        return self.vocab_size


def generate_tgt_mask(seq_len):
    """生成上三角的掩蔽矩阵，防止看到未来的词"""
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    return mask