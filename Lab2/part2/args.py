import argparse
import torch
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SparseMoE Transformer model for text generation.")
    parser.add_argument('--data_path', type=str, default='data/input.txt', help='Path to the input text file.')
    parser.add_argument('--chunk_size', type=int, default=50, help='Size of text chunks.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length for the model.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for the model.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in the Transformer.')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in SparseMoE.')
    parser.add_argument('--active_experts', type=int, default=2, help='Number of active experts in SparseMoE.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--save_path', type=str, default='results/', help='Path to save the model and results.')
    parser.add_argument('--model_path', type=str, default='models/', help='Path to save the model.')
    return parser.parse_args()


args = parse_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

args.tokenizer_mode = 'bert'

args.save_path = f"{args.save_path}/{args.tokenizer_mode}"
args.model_path = f"{args.model_path}/{args.tokenizer_mode}"

# mkdir model path
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)