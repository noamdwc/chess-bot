import torch
import wandb
import matplotlib.pyplot as plt
from optuna.integration.wandb import WeightsAndBiasesCallback

def save_checkpoint(model, model_path, loss, best_loss):
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), model_path)
        print(f"Model saved with loss: {loss}")
    return best_loss


def load_checkpoint(model, model_path):
    """
    Load model weights from the specified checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        model_path (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    return model




def split_tensor(batch_tensor):
    """
    Splits a tensor of shape (N, 774) into two tensors:
    - Board tensor of shape (N, 8, 8, 12)
    - Metadata tensor of shape (N, 6)

    Args:
        batch_tensor (torch.Tensor): Input tensor of shape (N, 774)

    Returns:
        board_tensor (torch.Tensor): Tensor of shape (N, 8, 8, 12)
        metadata_tensor (torch.Tensor): Tensor of shape (N, 6)
    """
    # Split the tensor into board data (first 768 values) and metadata (last 6 values)
    board_flat = batch_tensor[:, :768]  # Shape: (N, 768)
    metadata = batch_tensor[:, 768:]   # Shape: (N, 6)

    # Reshape the board data into (N, 8, 8, 12)
    board_tensor = board_flat.view(-1, 12, 8, 8)#.permute(0, 2, 3, 1)  # Shape: (N, 8, 8, 12)

    return board_tensor, metadata




def evaluate_model(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            features, target = batch
            features = features.to(device)
            target = target.to(device)
            board, metadata = split_tensor(features)
            output = model(board, metadata)
            loss = criterion(output, target.view(-1, 1))
            total_loss += loss.item()
    return total_loss / len(dataloader)



def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    return total_params, trainable_params, nontrainable_params

def predict_evaluation(input_1d):
    model.eval()
    with torch.no_grad():
        board, metadata = split_tensor(input_1d.unsqueeze(0))
        board = board.to(device)
        metadata = metadata.to(device)
        output = model(board, metadata)
    return output.item()




def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, schedualer, run_name=None):
    print(run_name)
    with wandb.init(project=project_name, name=run_name) as run:
      run.watch(model, criterion, log="all", log_freq=100)

      best_loss = float('inf')
      # Define a model path for saving the best checkpoint.
      # Use run_name to create a unique filename if provided.
      model_path = encoded_data_path + f"/{run_name}_best_model.pth" if run_name else "/best_model.pth"
      try:
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for i, batch in enumerate(train_loader):
                features, target = batch
                features = features.to(device)
                target = target.to(device)
                board, metadata = split_tensor(features)

                output = model(board, metadata)
                loss = criterion(output, target.view(-1, 1))
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if i % 500 == 0:
                    print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}")
                    run.log({"batch loss": loss.item(),
                              'epoch': epoch+1,
                              'batch': i + 1})

            train_loss = total_loss / len(train_loader)
            train_losses.append(train_loss)
            val_loss = evaluate_model(model, criterion, val_loader, device)
            val_losses.append(val_loss)
            schedualer.step(val_loss)

            # Save checkpoint if validation loss improved
            best_loss = save_checkpoint(model, model_path, val_loss, best_loss)

            print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            run.log({"train loss": train_loss,
                      "val loss": val_loss,
                      'epoch': epoch+1})
      except Exception as e:
        wandb.log({'error': str(e)})
        raise e
      return model, train_losses, val_losses


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()





wandb_callback = WeightsAndBiasesCallback(
    metric_name="val_loss",  # Match the metric used during training
    wandb_kwargs={"project": "chess-bot"}  # Set your W&B project
)

def objective(trial):
    emd_dim = trial.suggest_categorical("emd_dim", [16, 32, 64])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    nlayer = trial.suggest_int("nlayer", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    patience = trial.suggest_int("patience", 3, 10)

    model = ChessTransformer(
        emd_dim=emd_dim,
        nhead=nhead,
        nlayer=nlayer,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    # Training
    model, train_losses, val_losses = train_model(
        model,
        train_dataloder,
        val_dataloder,
        criterion,
        optimizer,
        10,
        scheduler
    )
    best_val_loss = min(val_losses)
    return best_val_loss