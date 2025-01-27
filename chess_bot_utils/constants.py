import torch

data_path = "/root/.cache/kagglehub/datasets/ronakbadhe/chess-evaluations/versions/5/chessData.csv"
encoded_data_path = 'drive/MyDrive/data/chess_bot_batches'
project_name = 'chess-bot'
light_encoded_path = f'{encoded_data_path}/light_774_1d'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

study_name = "chess-bot-opt-encoder-2heads"
storage_url = f"sqlite:///{encoded_data_path}/{study_name}.db"