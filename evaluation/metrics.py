import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_classification(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None,
                               task: str = 'binary') -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            task: 'binary' or 'multiclass'
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if task == 'binary':
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        else:
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Probability-based metrics
        if y_proba is not None:
            metrics['log_loss'] = log_loss(y_true, y_proba)
            
            if task == 'binary':
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['brier_score'] = brier_score_loss(y_true, y_proba[:, 1])
            else:
                # Multiclass AUC (one-vs-rest)
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_proba, 
                        multi_class='ovr', 
                        average='macro'
                    )
                except ValueError:
                    metrics['roc_auc_ovr'] = np.nan
            
            # Calibration error
            metrics['ece'] = self.expected_calibration_error(y_true, y_proba, task=task)
        
        return metrics
    
    def evaluate_regression(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Mean absolute percentage error
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.nan
        
        return metrics
    
    def expected_calibration_error(self,
                                  y_true: np.ndarray,
                                  y_proba: np.ndarray,
                                  n_bins: int = 10,
                                  task: str = 'binary') -> float:
        """
        Calculate Expected Calibration Error (ECE)
        Measures how well predicted probabilities match actual outcomes
        """
        if task == 'binary':
            y_prob = y_proba[:, 1]
        else:
            # For multiclass, use max probability
            y_prob = np.max(y_proba, axis=1)
            y_true_binary = (y_true == np.argmax(y_proba, axis=1)).astype(int)
            y_true = y_true_binary
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_prob[mask].mean()
                bin_weight = mask.sum() / len(y_true)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def plot_calibration_curve(self,
                              y_true: np.ndarray,
                              y_proba: np.ndarray,
                              n_bins: int = 10,
                              save_path: Optional[str] = None):
        """Plot reliability diagram"""
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba[:, 1], n_bins=n_bins, strategy='uniform'
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(prob_pred, prob_true, 'o-', label='Model')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             labels: Optional[list] = None,
                             save_path: Optional[str] = None):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def simulate_betting_performance(self,
                                    y_true: np.ndarray,
                                    y_proba: np.ndarray,
                                    odds: Optional[np.ndarray] = None,
                                    kelly_fraction: float = 0.25) -> Dict[str, float]:
        """
        Simulate betting performance using Kelly Criterion
        
        Args:
            y_true: True outcomes (0=home win, 1=away win, 2=OT)
            y_proba: Predicted probabilities
            odds: Betting odds (if None, use fair odds from predictions)
            kelly_fraction: Fraction of Kelly bet size (for safety)
        
        Returns:
            Betting performance metrics
        """
        if odds is None:
            # Use implied fair odds from predictions
            odds = 1 / y_proba
        
        bankroll = 1000.0  # Starting bankroll
        bankroll_history = [bankroll]
        
        for i in range(len(y_true)):
            # Find best bet (highest edge)
            implied_probs = 1 / odds[i]
            edges = y_proba[i] - implied_probs
            
            best_outcome = np.argmax(edges)
            edge = edges[best_outcome]
            
            if edge > 0:
                # Kelly criterion: f = (bp - q) / b
                # where b = odds-1, p = predicted prob, q = 1-p
                b = odds[i, best_outcome] - 1
                p = y_proba[i, best_outcome]
                q = 1 - p
                
                kelly_bet = (b * p - q) / b
                kelly_bet = max(0, kelly_bet)  # Don't bet if negative
                
                # Fractional Kelly for safety
                bet_size = kelly_fraction * kelly_bet * bankroll
                bet_size = min(bet_size, bankroll * 0.05)  # Max 5% of bankroll
                
                # Place bet
                if y_true[i] == best_outcome:
                    # Win
                    bankroll += bet_size * (odds[i, best_outcome] - 1)
                else:
                    # Lose
                    bankroll -= bet_size
            
            bankroll_history.append(bankroll)
        
        metrics = {
            'final_bankroll': bankroll,
            'roi': ((bankroll - 1000) / 1000) * 100,
            'max_bankroll': max(bankroll_history),
            'min_bankroll': min(bankroll_history),
            'num_bets': len(bankroll_history) - 1
        }
        
        return metrics
    
    def generate_evaluation_report(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_proba: Optional[np.ndarray] = None,
                                  task: str = 'multiclass',
                                  model_name: str = 'Model') -> str:
        """Generate comprehensive evaluation report"""
        
        report = f"\n{'='*80}\n"
        report += f"EVALUATION REPORT: {model_name}\n"
        report += f"{'='*80}\n\n"
        
        # Classification metrics
        metrics = self.evaluate_classification(y_true, y_pred, y_proba, task)
        
        report += "Classification Metrics:\n"
        report += f"  Accuracy: {metrics['accuracy']:.4f}\n"
        
        if 'precision_macro' in metrics:
            report += f"  Precision (macro): {metrics['precision_macro']:.4f}\n"
            report += f"  Recall (macro): {metrics['recall_macro']:.4f}\n"
            report += f"  F1 (macro): {metrics['f1_macro']:.4f}\n"
        
        if 'log_loss' in metrics:
            report += f"\nProbability Metrics:\n"
            report += f"  Log Loss: {metrics['log_loss']:.4f}\n"
            report += f"  Brier Score: {metrics.get('brier_score', 'N/A')}\n"
            report += f"  Expected Calibration Error: {metrics['ece']:.4f}\n"
        
        if 'roc_auc_ovr' in metrics and not np.isnan(metrics['roc_auc_ovr']):
            report += f"  ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}\n"
        
        report += f"\n{'='*80}\n"
        
        logger.info(report)
        return report