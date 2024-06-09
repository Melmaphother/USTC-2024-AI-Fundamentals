import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SparseMoE Transformer model for text generation.")
    parser.add_argument('--datapath', type=str, default='data/input.txt', help='Path to the input text file.')
    parser.add_argument('--chunk_size', type=int, default=50, help='Size of text chunks.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length for the model.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for the model.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in the Transformer.')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in SparseMoE.')
    parser.add_argument('--active_experts', type=int, default=2, help='Number of active experts in SparseMoE.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    return parser.parse_args()


args = parse_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

args.tokenizer_mode = 'bert'
