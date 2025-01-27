import torch
import torch.nn as nn
import torch.nn.functional as F

from training_utils import load_checkpoint, save_checkpoint

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        # Convolutional layers for board and en passant
        self.conv = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=1),  # Input: 12x8x8, Output: 16x8x8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Output: 32x8x8
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.metadata_projection = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.croos_attention = nn.MultiheadAttention(embed_dim=32, num_heads=8, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict evaluation score
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, board, metadata):
        batch_size = board.size(0)
        board_features = self.conv(board) # shape Nx32x8x8

        board_projection = board_features.permute(0, 2, 3, 1).reshape(batch_size, 64, 32)
        metadata = self.metadata_projection(metadata) # shape Nx32

        query = metadata.unsqueeze(1) # shape Nx1x32
        key = board_projection # shape Nx64x32
        value = board_projection # shape Nx64x32
        cross_attention_output, _ = self.croos_attention(query, key, value) # shape Nx1x128
        cross_attention_output = cross_attention_output.squeeze(1) # shape Nx128
        combined_features = torch.cat((cross_attention_output, metadata), dim=1) # shape Nx160
        combined_features = self.dropout(combined_features)

        # Pass through fully connected layers
        output = self.fc(combined_features)
        return output






class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add skip connection
        return F.relu(out)


class ChessResCNN(nn.Module):
    def __init__(self):
        super(ChessResCNN, self).__init__()
        # Convolutional layers for board and en passant
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(32) for _ in range(4)]
        )
        self.conv = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=1),  # Input: 12x8x8, Output: 16x8x8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Output: 32x8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.res_blocks
        )



        self.metadata_projection = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.croos_attention = nn.MultiheadAttention(embed_dim=32, num_heads=8, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict evaluation score
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, board, metadata):
        batch_size = board.size(0)
        board_features = self.conv(board) # shape Nx32x8x8

        board_projection = board_features.permute(0, 2, 3, 1).reshape(batch_size, 64, 32)
        metadata = self.metadata_projection(metadata) # shape Nx32

        query = metadata.unsqueeze(1) # shape Nx1x32
        key = board_projection # shape Nx64x32
        value = board_projection # shape Nx64x32
        cross_attention_output, _ = self.croos_attention(query, key, value) # shape Nx1x128
        cross_attention_output = cross_attention_output.squeeze(1) # shape Nx128
        combined_features = torch.cat((cross_attention_output, metadata), dim=1) # shape Nx160
        combined_features = self.dropout(combined_features)

        # Pass through fully connected layers
        output = self.fc(combined_features)
        return output





class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add skip connection
        return F.relu(out)




class ChessTransformer(nn.Module):
    def __init__(self, emd_dim, nhead, nlayer, dropout):
        super(ChessTransformer, self).__init__()
        self.emd_dim = emd_dim
        # Board encoder
        num_squres = 8 * 8
        # self.squre_embedding = nn.Linear(12, emd_dim)
        self.board_embedding = nn.Sequential(
            nn.Conv2d(12, emd_dim // 2, kernel_size=3, padding=1),  # Input: 12x8x8, Output: 16x8x8
            nn.BatchNorm2d(emd_dim // 2),
            nn.ReLU(),
            nn.Conv2d(emd_dim // 2, emd_dim, kernel_size=3, padding=1),  # Output: 32x8x8
            nn.BatchNorm2d(emd_dim),
            nn.ReLU(),
            *[ResidualBlock(emd_dim) for _ in range(16)],

        )
        self.position_embedding = nn.Parameter(torch.rand(1, num_squres, emd_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emd_dim,
                                                        nhead=nhead,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayer)


        self.metadata_projection = nn.Sequential(
            nn.Linear(6, emd_dim),
            nn.BatchNorm1d(emd_dim),
            nn.ReLU()
        )

        self.croos_attention = nn.MultiheadAttention(embed_dim=emd_dim, num_heads=8, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(emd_dim * 2, emd_dim),
            nn.BatchNorm1d(emd_dim),
            nn.ReLU(),
            nn.Linear(emd_dim, 1)  # Predict evaluation score
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, board, metadata):
        batch_size = board.size(0)
        board_features = self.board_embedding(board) # shape Nx32x8x8
        board_features = board_features.view(batch_size, 8*8, self.emd_dim) # shape Nx8*8(64)xEMB
        board_features = board_features + self.position_embedding
        board_features = self.transformer_encoder(board_features)

        metadata = self.metadata_projection(metadata) # shape NxEMB

        query = metadata.unsqueeze(1) # shape Nx1x32
        cross_attention_output, _ = self.croos_attention(query, board_features, board_features) # shape Nx1x128
        cross_attention_output = cross_attention_output.squeeze(1) # shape NxEMB
        combined_features = torch.cat((cross_attention_output, metadata), dim=1) # shape NxEMB*2
        combined_features = self.dropout(combined_features)

        # Pass through fully connected layers
        output = self.fc(combined_features)
        return output



class ResidualBlockIsMates(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super(ResidualBlockIsMates, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # Add skip connection
        return F.relu(out)


class IsMateRes(nn.Module):
    def __init__(self):
        super(IsMateRes, self).__init__()
        # Convolutional layers for board and en passant
        self.res_blocks = nn.Sequential(
            *[ResidualBlockIsMates(16, 0.2) for _ in range(4)]
        )
        self.conv = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=1),  # Input: 12x8x8, Output: 16x8x8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            self.res_blocks
        )



        self.metadata_projection = nn.Sequential(
            nn.Linear(6, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.croos_attention = nn.MultiheadAttention(embed_dim=16, num_heads=8, batch_first=True)

        # Fully connected layers
        self.fc = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, board, metadata):
        batch_size = board.size(0)
        board_features = self.conv(board) # shape Nx16x8x8

        board_projection = board_features.permute(0, 2, 3, 1).reshape(batch_size, 64, 16)
        metadata = self.metadata_projection(metadata) # shape Nx16

        query = metadata.unsqueeze(1) # shape Nx1x16
        key = board_projection # shape Nx64x16
        value = board_projection # shape Nx64x16
        cross_attention_output, _ = self.croos_attention(query, key, value) # shape Nx1x32
        cross_attention_output = cross_attention_output.squeeze(1) # shape Nx32
        combined_features = torch.cat((cross_attention_output, metadata), dim=1) # shape Nx48
        combined_features = self.dropout(combined_features)

        # Pass through fully connected layers
        output = self.fc(combined_features)
        return output



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add skip connection
        return F.relu(out)


class ChessEncoder(nn.Module):
    def __init__(self, dropout, num_res_blocks):
        super(ChessEncoder, self).__init__()
        self.res_blocks64 = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)],
            nn.Dropout2d(dropout)
        )


        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),  # Input: 12x8x8, Output: 16x8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: 32x8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )



        self.metadata_projection = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.croos_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)


    def forward(self, board, metadata):
        metadata_projection = self.metadata_projection(metadata) # shape Nx64
        batch_size = board.size(0)
        board_features = self.conv1(board)
        board_projection = board_features.permute(0, 2, 3, 1).reshape(batch_size, 64, 64) # shape of BATCHx(8*8)xCHANELS
        query = metadata_projection.unsqueeze(1) # shape Nx1x64
        key = board_projection # shape Nx64x64
        value = board_projection # shape Nx64x64
        cross_attention_output, _ = self.croos_attention(query, key, value) # shape Nx1x64
        cross_attention_output = cross_attention_output.squeeze(1).unsqueeze(2).unsqueeze(3) # shape Nx64x1x1
        board_attn_features = board_features + cross_attention_output # shape Nx64x8x8
        board_features = self.res_blocks64(board_attn_features)
        return board_features, metadata_projection


class ChessResHead(nn.Module):
    def __init__(self, encoder, dropout, num_res_blocks):
        super(ChessResHead, self).__init__()
        self.encoder = encoder

        self.res_blocks128 = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_res_blocks)],
            nn.Dropout2d(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, padding=0),  # Input: 64x8x8, Output: 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0),  # Input: 128x8x8, Output: 64x8x8
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Converts (B, 64, 8, 8) -> (B, 64, 1, 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict evaluation score
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, board, metadata):
        # shape Nx64x8x8, shape Nx64
        board_features, metadata_projection = self.encoder(board, metadata)

        board_features = self.conv2(board_features) # shape Nx128x8
        board_features = self.res_blocks128(board_features)
        board_features = self.conv3(board_features) # shape Nx64x8x8
        board_features = self.gap(board_features) # shape Nx64x1x1
        board_features = board_features.squeeze(2).squeeze(2) # shape Nx64

        combined_features = torch.cat((board_features, metadata_projection), dim=1) # shape Nx128
        combined_features = self.dropout(combined_features)

        # Pass through fully connected layers
        output = self.fc(combined_features)
        return output



class ChessVITHead(nn.Module):
  def __init__(self, encoder, dropout, num_res_blocks, embed_dim=128, nhead=8, cross_nhead=8):
    super(ChessVITHead, self).__init__()
    self.encoder = encoder
    self.patch_ecoder = nn.Conv2d(64, embed_dim, kernel_size=1, padding=0)
    self.flatten = nn.Sequential(
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(start_dim=2)
    )
    self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    self.positional_embedding = nn.Parameter(torch.randn(1, 64 + 1, embed_dim))

    encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_res_blocks)

    self.metadata_projection = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
    self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=cross_nhead, batch_first=True)


    self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict evaluation score
        )


class CombinedModel(nn.Module):
    def __init__(self, mate_model, no_mate_model, is_mate_model, is_mate_threshold=0.5):
        super(CombinedModel, self).__init__()
        self.mate_model = mate_model
        self.no_mate_model = no_mate_model
        self.is_mate_model = is_mate_model
        self.is_mate_model.requires_grad_(False)
        self.is_mate_model.eval()
        self.is_mate_threshold = is_mate_threshold

    def forward(self, board, metadata):
        is_mate_output = self.is_mate_model(board, metadata)
        is_mate_mask = (torch.sigmoid(is_mate_output) > self.is_mate_threshold).float()
        mate_output = self.mate_model(board, metadata)
        no_mate_output = self.no_mate_model(board, metadata)
        combined_output = torch.where(is_mate_mask == 1, mate_output, no_mate_output)
        return combined_output

    @staticmethod
    def create_combined_model(enc_num_res_blocks,
                              encoder_dropout,
                              mate_head_num_res_blocks,
                              mate_head_dropout,
                              no_mate_head_num_res_blocks,
                              no_mate_head_dropout,
                              encoded_data_path,
                              device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = ChessEncoder(encoder_dropout, enc_num_res_blocks).to(device)
        mate_model = ChessResHead(encoder, mate_head_dropout, mate_head_num_res_blocks).to(device)
        no_mate_model = ChessResHead(encoder, no_mate_head_dropout, no_mate_head_num_res_blocks).to(device)
        is_mate_model = load_checkpoint(IsMateRes(), encoded_data_path + '/is_mate_best_model.pth')
        is_mate_model = is_mate_model.to(device)
        combined_model = CombinedModel(mate_model, no_mate_model, is_mate_model).to(device)
        return combined_model
