import os

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import data_path, light_encoded_path


def get_data():
    if not os.path.exists(data_path):
        path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
    return pd.read_csv(data_path)


def encode_evaluation(evaluation_str):
    """
    Encode chess evaluation string to a numeric value.

    Parameters:
        evaluation_str (str): Evaluation string (e.g., '+3.5', '#+6', '#-3').

    Returns:
        float: Encoded numeric value.
    """
    if evaluation_str.startswith('#'):
        # Mate evaluation
        mate_value = int(evaluation_str[2:])  # Extract the number (e.g., '6' from '#+6')
        if evaluation_str[1] == '+':
            return (100 - mate_value) / 100  # White mates in mate_value plies, normalized
        elif evaluation_str[1] == '-':
            return (-100 + mate_value) / 100  # Black mates in mate_value plies, normalized
    else:
        # Numeric evaluation
        return float(evaluation_str) / 100  # Normalize by 100


class ChessDataset(Dataset):
    def __init__(self, fens=None, evaluations=None, preprocessed_file=None):
        """
        Initialize the dataset. If a preprocessed file is provided, load the data from it.
        Otherwise, encode the FEN strings and evaluations and save them for future use.

        Args:
            fens (list of str): List of FEN strings.
            evaluations (list of float): Corresponding evaluation scores.
            preprocessed_file (str): Path to the preprocessed dataset file.
        """
        if preprocessed_file and os.path.exists(preprocessed_file):
            print(f"Loading preprocessed dataset from {preprocessed_file}...")
            data = torch.load(preprocessed_file)
            self.data = data['data']
            self.labels = data['labels']
        elif fens and evaluations:
            self.data = [self.fen_to_vector(fen) for fen in tqdm(fens)]
            self.labels = torch.tensor(evaluations, dtype=torch.float32)
            if preprocessed_file:
                print(f"Saving preprocessed dataset to {preprocessed_file}...")
                torch.save({'data': self.data, 'labels': self.labels}, preprocessed_file)
        else:
            raise ValueError("Either provide fens and evaluations or a preprocessed file.")

    @staticmethod
    def fen_to_vector(fen):
        """
        Converts a FEN string to a flat numeric vector representation.
        """
        piece_to_plane = {
            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,  # Black pieces
            'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,  # White pieces
        }
        tensor = np.zeros((12, 8, 8), dtype=np.int8)
        parts = fen.split()

        # Parse the board position
        rows = parts[0].split('/')
        for rank, row in enumerate(rows):
            file = 0
            for char in row:
                if char.isdigit():
                    file += int(char)  # Skip empty squares
                else:
                    plane = piece_to_plane[char]
                    tensor[plane, rank, file] = 1  # Mark the piece on its plane
                    file += 1

        # Flatten the board (8x8x12 -> 768)
        board_vector = tensor.flatten()

        # Parse active color
        active_color = [1] if parts[1] == 'w' else [0]

        # Parse castling rights
        castling_rights = parts[2]
        castling_vector = [
            1 if 'K' in castling_rights else 0,
            1 if 'Q' in castling_rights else 0,
            1 if 'k' in castling_rights else 0,
            1 if 'q' in castling_rights else 0,
        ]

        # Parse en passant target square
        en_passant = parts[3]
        if en_passant == '-':
            en_passant_file = [-1]  # No en passant
        else:
            en_passant_file = [ord(en_passant[0]) - ord('a')]

        # Combine into a single vector
        return torch.tensor(
            np.concatenate([board_vector, active_color, castling_vector, en_passant_file]),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def encode_and_create(df, save_dir):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    train_dataset = ChessDataset(train_df['FEN'].tolist(), train_df['Evaluation'].tolist(), save_dir + '/train.pt')
    val_dataset = ChessDataset(val_df['FEN'].tolist(), val_df['Evaluation'].tolist(), light_encoded_path + '/val.pt')
    test_dataset = ChessDataset(test_df['FEN'].tolist(), test_df['Evaluation'].tolist(),
                                light_encoded_path + '/test.pt')
    return train_dataset, val_dataset, test_dataset
