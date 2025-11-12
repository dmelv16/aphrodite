import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class NHLDataset(Dataset):
    """PyTorch Dataset for NHL game data"""
    
    def __init__(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        self.X = torch.FloatTensor(X)
        self.y_dict = {k: torch.FloatTensor(v) for k, v in y_dict.items()}
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.y_dict.items()}


class MultiTaskNHLModel(nn.Module):
    """
    Multi-task neural network for NHL prediction
    Jointly predicts: game outcome, home score, away score
    """
    
    def __init__(self,
                 input_dim: int,
                 shared_layers: List[int] = [512, 256, 128],
                 task_heads: Dict[str, List[int]] = None,
                 dropout: float = 0.3):
        super(MultiTaskNHLModel, self).__init__()
        
        if task_heads is None:
            task_heads = {
                'outcome': [128, 64],
                'home_score': [64],
                'away_score': [64]
            }
        
        # Shared encoder layers
        shared_modules = []
        prev_dim = input_dim
        
        for hidden_dim in shared_layers:
            shared_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*shared_modules)
        
        # Task-specific heads
        self.outcome_head = self._build_head(prev_dim, task_heads['outcome'], output_dim=3, dropout=dropout)
        self.home_score_head = self._build_head(prev_dim, task_heads['home_score'], output_dim=1, dropout=dropout)
        self.away_score_head = self._build_head(prev_dim, task_heads['away_score'], output_dim=1, dropout=dropout)
    
    def _build_head(self, input_dim: int, hidden_layers: List[int], output_dim: int, dropout: float):
        """Build task-specific head"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        # Shared representation
        shared_repr = self.shared_encoder(x)
        
        # Task-specific predictions
        outcome_logits = self.outcome_head(shared_repr)  # 3 classes: home win, away win, OT
        home_score = torch.relu(self.home_score_head(shared_repr))  # Non-negative
        away_score = torch.relu(self.away_score_head(shared_repr))
        
        return {
            'outcome': outcome_logits,
            'home_score': home_score.squeeze(),
            'away_score': away_score.squeeze()
        }


class MultiTaskTrainer:
    """Trainer for multi-task NHL model with GPU support"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 task_weights: Optional[Dict[str, float]] = None):
        
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        
        # Default task weights
        if task_weights is None:
            task_weights = {
                'outcome': 1.0,
                'home_score': 0.5,
                'away_score': 0.5
            }
        self.task_weights = task_weights
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss functions
        self.outcome_criterion = nn.CrossEntropyLoss()
        self.score_criterion = nn.MSELoss()
        
        # For mixed precision training
        self.scaler_amp = torch.cuda.amp.GradScaler()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_outcome_loss': [],
            'train_score_loss': []
        }
        
        logger.info(f"Model initialized on {device}")
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute multi-task loss"""
        
        # Outcome loss (classification)
        outcome_loss = self.outcome_criterion(predictions['outcome'], targets['outcome'].long())
        
        # Score losses (regression with Poisson-like characteristics)
        home_score_loss = self.score_criterion(predictions['home_score'], targets['home_score'])
        away_score_loss = self.score_criterion(predictions['away_score'], targets['away_score'])
        score_loss = (home_score_loss + away_score_loss) / 2
        
        # Combined weighted loss
        total_loss = (
            self.task_weights['outcome'] * outcome_loss +
            self.task_weights['home_score'] * home_score_loss +
            self.task_weights['away_score'] * away_score_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'outcome': outcome_loss.item(),
            'score': score_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, dataloader: DataLoader, use_amp: bool = True) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'outcome': 0, 'score': 0}
        
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(self.device)
            y_batch = {k: v.to(self.device) for k, v in y_batch.items()}
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if use_amp and self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    predictions = self.model(X_batch)
                    loss, loss_dict = self.compute_loss(predictions, y_batch)
                
                self.scaler_amp.scale(loss).backward()
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()
            else:
                predictions = self.model(X_batch)
                loss, loss_dict = self.compute_loss(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
            
            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_losses[k] += v
        
        # Average losses
        n_batches = len(dataloader)
        return {k: v / n_batches for k, v in epoch_losses.items()}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = {'total': 0, 'outcome': 0, 'score': 0}
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = {k: v.to(self.device) for k, v in y_batch.items()}
                
                predictions = self.model(X_batch)
                loss, loss_dict = self.compute_loss(predictions, y_batch)
                
                for k, v in loss_dict.items():
                    val_losses[k] += v
        
        n_batches = len(dataloader)
        return {k: v / n_batches for k, v in val_losses.items()}
    
    def fit(self,
            X_train: np.ndarray,
            y_train: Dict[str, np.ndarray],
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[Dict[str, np.ndarray]] = None,
            epochs: int = 100,
            batch_size: int = 1024,
            patience: int = 15,
            use_amp: bool = True) -> Dict:
        """
        Train model with early stopping
        
        Args:
            X_train: Training features
            y_train: Dict with keys 'outcome', 'home_score', 'away_score'
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            use_amp: Use automatic mixed precision
        """
        
        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)
        
        # Create datasets
        train_dataset = NHLDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        if X_val is not None:
            val_dataset = NHLDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device == 'cuda' else False
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(train_loader, use_amp=use_amp)
            
            # Validate
            if X_val is not None:
                val_losses = self.validate(val_loader)
                self.history['val_loss'].append(val_losses['total'])
                
                # Early stopping
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses['total']:.4f}, Val Loss: {val_losses['total']:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses['total']:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_outcome_loss'].append(train_losses['outcome'])
            self.history['train_score_loss'].append(train_losses['score'])
            
            # Step scheduler
            self.scheduler.step()
        
        return self.history
    
    def predict(self, X: np.ndarray, batch_size: int = 1024) -> Dict[str, np.ndarray]:
        """Make predictions"""
        self.model.eval()
        
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X)
        
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = {'outcome': [], 'home_score': [], 'away_score': []}
        
        with torch.no_grad():
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                
                # Convert outcome to probabilities
                outcome_probs = torch.softmax(pred['outcome'], dim=1)
                
                predictions['outcome'].append(outcome_probs.cpu().numpy())
                predictions['home_score'].append(pred['home_score'].cpu().numpy())
                predictions['away_score'].append(pred['away_score'].cpu().numpy())
        
        return {
            'outcome': np.vstack(predictions['outcome']),
            'home_score': np.concatenate(predictions['home_score']),
            'away_score': np.concatenate(predictions['away_score'])
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler_amp.state_dict(),
            'history': self.history
        }, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler_amp.load_state_dict(checkpoint['scaler_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {filepath}")