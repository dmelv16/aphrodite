"""
Poisson-based Multi-task Neural Network for NHL Predictions
Predicts lambda parameters for exact goal probability distributions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class NHLPoissonDataset(Dataset):
    """PyTorch Dataset for NHL game data"""
    
    def __init__(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        self.X = torch.FloatTensor(X)
        
        # Store targets with appropriate types
        self.y_dict = {
            'outcome': torch.LongTensor(y_dict['outcome']),  # Classification
            'home_score': torch.FloatTensor(y_dict['home_score']),  # Count data for Poisson
            'away_score': torch.FloatTensor(y_dict['away_score'])   # Count data for Poisson
        }
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.y_dict.items()}


class PoissonMultiTaskNHLModel(nn.Module):
    """
    IMPROVED: Multi-task neural network with Poisson outputs for score prediction
    
    Key changes:
    - Outputs lambda parameters for Poisson distributions
    - Uses Softplus activation to ensure positive lambdas
    - Enables exact probability calculations for any score
    """
    
    def __init__(self,
                 input_dim: int,
                 shared_layers: List[int] = [512, 256, 128],
                 task_heads: Dict[str, List[int]] = None,
                 dropout: float = 0.3):
        super(PoissonMultiTaskNHLModel, self).__init__()
        
        if task_heads is None:
            task_heads = {
                'outcome': [128, 64],
                'home_score': [64, 32],  # Deeper for Poisson parameter learning
                'away_score': [64, 32]
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
        self.outcome_head = self._build_classification_head(
            prev_dim, task_heads['outcome'], output_dim=3, dropout=dropout
        )
        
        # CRITICAL CHANGE: Poisson heads with Softplus activation
        self.home_score_head = self._build_poisson_head(
            prev_dim, task_heads['home_score'], dropout=dropout
        )
        self.away_score_head = self._build_poisson_head(
            prev_dim, task_heads['away_score'], dropout=dropout
        )
        
        # Softplus for ensuring positive lambda values
        self.softplus = nn.Softplus()
        
    def _build_classification_head(self, input_dim: int, hidden_layers: List[int], 
                                  output_dim: int, dropout: float):
        """Build classification head for outcome prediction"""
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
    
    def _build_poisson_head(self, input_dim: int, hidden_layers: List[int], dropout: float):
        """
        Build Poisson parameter prediction head
        No final activation here - we apply Softplus in forward()
        """
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final linear layer outputs unconstrained values
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass - outputs lambda parameters for Poisson distributions
        """
        # Shared representation
        shared_repr = self.shared_encoder(x)
        
        # Task-specific predictions
        outcome_logits = self.outcome_head(shared_repr)  # 3 classes
        
        # CRITICAL: Output Poisson lambda parameters (always positive)
        home_lambda_raw = self.home_score_head(shared_repr)
        away_lambda_raw = self.away_score_head(shared_repr)
        
        # Apply Softplus to ensure λ > 0
        # Adding small epsilon for numerical stability
        home_lambda = self.softplus(home_lambda_raw).squeeze() + 1e-8
        away_lambda = self.softplus(away_lambda_raw).squeeze() + 1e-8
        
        return {
            'outcome': outcome_logits,
            'home_lambda': home_lambda,  # Poisson rate parameter
            'away_lambda': away_lambda   # Poisson rate parameter
        }


class PoissonMultiTaskTrainer:
    """
    IMPROVED: Trainer using Poisson NLL Loss for score predictions
    """
    
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
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Loss functions
        self.outcome_criterion = nn.CrossEntropyLoss()
        
        # CRITICAL CHANGE: Use Poisson NLL Loss for scores
        self.poisson_criterion = nn.PoissonNLLLoss(log_input=False, full=True)
        
        # For mixed precision training
        self.scaler_amp = torch.cuda.amp.GradScaler()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_outcome_loss': [],
            'train_home_poisson_loss': [],
            'train_away_poisson_loss': []
        }
        
        logger.info(f"Poisson model initialized on {device}")
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-task loss with Poisson NLL for scores
        """
        
        # Outcome loss (classification)
        outcome_loss = self.outcome_criterion(predictions['outcome'], targets['outcome'])
        
        # CRITICAL: Poisson NLL losses for scores
        # The Poisson NLL expects:
        # - input: predicted lambda (rate parameter)
        # - target: actual count (goals scored)
        home_poisson_loss = self.poisson_criterion(
            predictions['home_lambda'], 
            targets['home_score']
        )
        away_poisson_loss = self.poisson_criterion(
            predictions['away_lambda'], 
            targets['away_score']
        )
        
        # Combined weighted loss
        total_loss = (
            self.task_weights['outcome'] * outcome_loss +
            self.task_weights['home_score'] * home_poisson_loss +
            self.task_weights['away_score'] * away_poisson_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'outcome': outcome_loss.item(),
            'home_poisson': home_poisson_loss.item(),
            'away_poisson': away_poisson_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'outcome': 0, 'home_poisson': 0, 'away_poisson': 0}
        
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(self.device)
            y_batch = {k: v.to(self.device) for k, v in y_batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_batch)
            loss, loss_dict = self.compute_loss(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
        val_losses = {'total': 0, 'outcome': 0, 'home_poisson': 0, 'away_poisson': 0}
        
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
    
    def predict_poisson_params(self, X: np.ndarray, batch_size: int = 1024) -> Dict[str, np.ndarray]:
        """
        Make predictions - returns Poisson lambda parameters
        
        Returns:
            Dict with 'outcome' probabilities and Poisson lambdas for scores
        """
        self.model.eval()
        
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X)
        
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = {
            'outcome': [], 
            'home_lambda': [], 
            'away_lambda': [],
            'home_probs': [],  # Full probability distributions
            'away_probs': []
        }
        
        with torch.no_grad():
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                
                # Convert outcome to probabilities
                outcome_probs = torch.softmax(pred['outcome'], dim=1)
                
                # Get lambda parameters
                home_lambdas = pred['home_lambda'].cpu().numpy()
                away_lambdas = pred['away_lambda'].cpu().numpy()
                
                predictions['outcome'].append(outcome_probs.cpu().numpy())
                predictions['home_lambda'].append(home_lambdas)
                predictions['away_lambda'].append(away_lambdas)
                
                # Calculate full probability distributions for scores 0-10
                for i, (h_lambda, a_lambda) in enumerate(zip(home_lambdas, away_lambdas)):
                    home_probs = [self.poisson_pmf(h_lambda, k) for k in range(11)]
                    away_probs = [self.poisson_pmf(a_lambda, k) for k in range(11)]
                    predictions['home_probs'].append(home_probs)
                    predictions['away_probs'].append(away_probs)
        
        return {
            'outcome': np.vstack(predictions['outcome']),
            'home_lambda': np.concatenate(predictions['home_lambda']),
            'away_lambda': np.concatenate(predictions['away_lambda']),
            'home_probs': np.array(predictions['home_probs']),
            'away_probs': np.array(predictions['away_probs'])
        }
    
    def poisson_pmf(self, lambda_param, k):
        """Calculate Poisson probability mass function"""
        return stats.poisson.pmf(k, lambda_param)
    
    def fit(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray],
            X_val: Optional[np.ndarray] = None, y_val: Optional[Dict[str, np.ndarray]] = None,
            epochs: int = 100, batch_size: int = 1024, patience: int = 15) -> Dict:
        """
        Train model with early stopping
        """
        
        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)
        
        # Create datasets
        train_dataset = NHLPoissonDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        if X_val is not None:
            val_dataset = NHLPoissonDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            if X_val is not None:
                val_losses = self.validate(val_loader)
                self.history['val_loss'].append(val_losses['total'])
                
                # Learning rate scheduler
                self.scheduler.step(val_losses['total'])
                
                # Early stopping
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint('best_poisson_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_losses['total']:.4f} "
                    f"(Outcome: {train_losses['outcome']:.4f}, "
                    f"Home Poisson: {train_losses['home_poisson']:.4f}, "
                    f"Away Poisson: {train_losses['away_poisson']:.4f}) - "
                    f"Val Loss: {val_losses['total']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_losses['total']:.4f}"
                )
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_outcome_loss'].append(train_losses['outcome'])
            self.history['train_home_poisson_loss'].append(train_losses['home_poisson'])
            self.history['train_away_poisson_loss'].append(train_losses['away_poisson'])
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state': self.scaler.__dict__,
            'history': self.history
        }, filepath)
        logger.info(f"Poisson model checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.__dict__ = checkpoint['scaler_state']
        self.history = checkpoint['history']
        logger.info(f"Poisson model loaded from {filepath}")


def analyze_poisson_predictions(trainer, X_test, y_test_home, y_test_away):
    """
    Analyze Poisson model predictions with exact probability calculations
    """
    # Get Poisson parameters
    predictions = trainer.predict_poisson_params(X_test)
    
    home_lambdas = predictions['home_lambda']
    away_lambdas = predictions['away_lambda']
    
    print("=" * 80)
    print("POISSON MODEL ANALYSIS")
    print("=" * 80)
    
    # 1. Lambda parameter analysis
    print("\n1. LAMBDA PARAMETER STATISTICS")
    print("-" * 40)
    print(f"Home Lambda - Mean: {home_lambdas.mean():.3f}, Std: {home_lambdas.std():.3f}")
    print(f"Away Lambda - Mean: {away_lambdas.mean():.3f}, Std: {away_lambdas.std():.3f}")
    print(f"Total Lambda - Mean: {(home_lambdas + away_lambdas).mean():.3f}")
    
    # 2. Compare predicted vs actual distributions
    print("\n2. SCORE DISTRIBUTION COMPARISON")
    print("-" * 40)
    
    # Calculate expected scores from lambdas
    expected_home = home_lambdas  # For Poisson, E[X] = λ
    expected_away = away_lambdas
    
    mae_home = np.mean(np.abs(expected_home - y_test_home))
    mae_away = np.mean(np.abs(expected_away - y_test_away))
    
    print(f"Home Score MAE: {mae_home:.3f}")
    print(f"Away Score MAE: {mae_away:.3f}")
    print(f"Total Score MAE: {np.mean(np.abs(expected_home + expected_away - (y_test_home + y_test_away))):.3f}")
    
    # 3. Calculate exact Over/Under probabilities
    print("\n3. EXACT OVER/UNDER PROBABILITIES")
    print("-" * 40)
    
    for line in [5.5, 6.5, 7.5]:
        over_probs = []
        
        for h_lambda, a_lambda in zip(home_lambdas, away_lambdas):
            # Calculate exact probability using Poisson
            prob_over = 0
            for h in range(11):  # Consider scores up to 10
                for a in range(11):
                    if h + a > line:
                        prob_over += stats.poisson.pmf(h, h_lambda) * stats.poisson.pmf(a, a_lambda)
            over_probs.append(prob_over)
        
        over_probs = np.array(over_probs)
        actual_over = (y_test_home + y_test_away > line).mean()
        
        print(f"\nLine {line}:")
        print(f"  Mean Predicted P(Over): {over_probs.mean():.3f}")
        print(f"  Actual Over Rate: {actual_over:.3f}")
        print(f"  Calibration Error: {over_probs.mean() - actual_over:+.3f}")
    
    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Lambda distributions
    axes[0, 0].hist(home_lambdas, bins=30, alpha=0.7, label='Home λ', color='blue')
    axes[0, 0].hist(away_lambdas, bins=30, alpha=0.7, label='Away λ', color='red')
    axes[0, 0].set_xlabel('Lambda (Poisson Rate)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Predicted Lambda Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Expected vs Actual scores
    axes[0, 1].scatter(y_test_home, expected_home, alpha=0.5, s=10, label='Home')
    axes[0, 1].scatter(y_test_away, expected_away, alpha=0.5, s=10, label='Away')
    axes[0, 1].plot([0, 8], [0, 8], 'k--', lw=2)
    axes[0, 1].set_xlabel('Actual Score')
    axes[0, 1].set_ylabel('Expected Score (λ)')
    axes[0, 1].set_title('Poisson Expected vs Actual Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Probability distribution for a sample game
    sample_idx = 0
    h_lambda = home_lambdas[sample_idx]
    a_lambda = away_lambdas[sample_idx]
    
    scores = range(8)
    home_probs = [stats.poisson.pmf(k, h_lambda) for k in scores]
    away_probs = [stats.poisson.pmf(k, a_lambda) for k in scores]
    
    x = np.arange(len(scores))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, home_probs, width, label='Home', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, away_probs, width, label='Away', color='red', alpha=0.7)
    axes[1, 0].axvline(y_test_home[sample_idx], color='blue', linestyle='--', label='Actual Home')
    axes[1, 0].axvline(y_test_away[sample_idx], color='red', linestyle='--', label='Actual Away')
    axes[1, 0].set_xlabel('Goals')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_title(f'Sample Game Probability Distribution\nλ_home={h_lambda:.2f}, λ_away={a_lambda:.2f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total score distribution
    total_probs = {}
    for h in range(11):
        for a in range(11):
            total = h + a
            prob = stats.poisson.pmf(h, h_lambda) * stats.poisson.pmf(a, a_lambda)
            if total in total_probs:
                total_probs[total] += prob
            else:
                total_probs[total] = prob
    
    totals = sorted(total_probs.keys())
    probs = [total_probs[t] for t in totals]
    
    axes[1, 1].bar(totals, probs, alpha=0.7, color='green')
    axes[1, 1].axvline(y_test_home[sample_idx] + y_test_away[sample_idx], 
                      color='red', linestyle='--', lw=2, label='Actual Total')
    axes[1, 1].set_xlabel('Total Goals')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_title(f'Total Score Distribution\nλ_total={h_lambda + a_lambda:.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('poisson_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return predictions


# Integration function for your existing pipeline
def upgrade_to_poisson_model(pipeline, X_train, y_train, X_val, y_val):
    """
    Upgrade existing pipeline to use Poisson model
    
    Args:
        pipeline: Your NHLPredictionPipeline instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
    """
    # Create Poisson model
    input_dim = X_train.shape[1]
    poisson_model = PoissonMultiTaskNHLModel(
        input_dim=input_dim,
        shared_layers=[512, 256, 128],
        dropout=0.3
    )
    
    # Create trainer
    trainer = PoissonMultiTaskTrainer(
        poisson_model,
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    # Prepare targets
    y_train_dict = {
        'outcome': y_train['outcome'],
        'home_score': y_train['home_score'],
        'away_score': y_train['away_score']
    }
    
    y_val_dict = {
        'outcome': y_val['outcome'],
        'home_score': y_val['home_score'],
        'away_score': y_val['away_score']
    }
    
    # Train
    print("Training Poisson Neural Network...")
    history = trainer.fit(
        X_train, y_train_dict,
        X_val, y_val_dict,
        epochs=50,
        batch_size=512,
        patience=10
    )
    
    # Store in pipeline
    pipeline.models['poisson_multitask'] = trainer
    
    print("\nPoisson model training complete!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    
    return trainer