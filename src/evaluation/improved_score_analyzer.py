"""
Enhanced NHL Score Prediction & Over/Under Analysis
With Poisson distributions, empirical noise estimation, and calibration analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')


class ImprovedScorePredictionAnalyzer:
    """
    Enhanced analyzer with Poisson-based probabilities and better calibration
    """
    
    def __init__(self, y_true_home, y_true_away, y_pred_home, y_pred_away, 
                 prediction_type='raw', model_outputs_log=False):
        """
        Initialize with true and predicted scores
        
        Args:
            y_true_home: Actual home team scores
            y_true_away: Actual away team scores  
            y_pred_home: Predicted home team scores (or lambda parameters)
            y_pred_away: Predicted away team scores (or lambda parameters)
            prediction_type: 'raw' for point estimates, 'poisson' for lambda parameters
            model_outputs_log: If True, predictions are log-transformed and need exp()
        """
        self.y_true_home = np.array(y_true_home)
        self.y_true_away = np.array(y_true_away)
        
        # Handle log-transformed predictions if necessary
        if model_outputs_log:
            self.y_pred_home = np.exp(y_pred_home)
            self.y_pred_away = np.exp(y_pred_away)
        else:
            self.y_pred_home = np.array(y_pred_home)
            self.y_pred_away = np.array(y_pred_away)
        
        self.prediction_type = prediction_type
        
        # Calculate totals
        self.y_true_total = self.y_true_home + self.y_true_away
        self.y_pred_total = self.y_pred_home + self.y_pred_away
        
        # CRITICAL FIX: Calculate empirical error statistics
        self.home_errors = self.y_pred_home - self.y_true_home
        self.away_errors = self.y_pred_away - self.y_true_away
        self.total_errors = self.y_pred_total - self.y_true_total
        
        # Empirical standard deviations for uncertainty estimation
        self.home_error_std = np.std(self.home_errors)
        self.away_error_std = np.std(self.away_errors)
        self.total_error_std = np.std(self.total_errors)
        
        print(f"Empirical Error Std Devs - Home: {self.home_error_std:.3f}, "
              f"Away: {self.away_error_std:.3f}, Total: {self.total_error_std:.3f}")
    
    def poisson_probability(self, lambda_param, k):
        """
        Calculate P(X=k) for Poisson distribution
        
        Args:
            lambda_param: Poisson rate parameter (can be array)
            k: Number of goals
        """
        return np.exp(-lambda_param) * (lambda_param ** k) / factorial(k)
    
    def calculate_total_distribution(self, lambda_home, lambda_away, max_goals=15):
        """
        Calculate exact probability distribution for total goals
        Assumes independence between home and away scores
        
        Args:
            lambda_home: Poisson parameter for home team
            lambda_away: Poisson parameter for away team
            max_goals: Maximum goals to consider for each team
            
        Returns:
            Dict with probabilities for each possible total
        """
        total_probs = {}
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                total = home_goals + away_goals
                prob = (self.poisson_probability(lambda_home, home_goals) * 
                       self.poisson_probability(lambda_away, away_goals))
                
                if total in total_probs:
                    total_probs[total] += prob
                else:
                    total_probs[total] = prob
        
        return total_probs
    
    def calculate_over_probability_poisson(self, line, game_idx):
        """
        Calculate exact Over probability using Poisson distributions
        
        Args:
            line: O/U line
            game_idx: Index of the game
            
        Returns:
            Probability of going OVER the line
        """
        if self.prediction_type != 'poisson':
            # Fall back to using predictions as lambda parameters
            lambda_home = self.y_pred_home[game_idx]
            lambda_away = self.y_pred_away[game_idx]
        else:
            lambda_home = self.y_pred_home[game_idx]
            lambda_away = self.y_pred_away[game_idx]
        
        # Get full distribution
        total_dist = self.calculate_total_distribution(lambda_home, lambda_away)
        
        # Sum probabilities for totals > line
        over_prob = sum(prob for total, prob in total_dist.items() if total > line)
        
        return over_prob
    
    def calculate_over_probability_empirical(self, line, noise_std=None):
        """
        IMPROVED: Calculate Over probability using empirical error distribution
        
        Args:
            line: The O/U line
            noise_std: If None, uses empirical std from test set
        
        Returns:
            Array of probabilities for each game
        """
        if noise_std is None:
            # USE ACTUAL OBSERVED ERROR - This is the key fix!
            noise_std = self.total_error_std
        
        # Calculate z-scores
        z_scores = (self.y_pred_total - line) / noise_std
        
        # Probability of going OVER
        probabilities = 1 - stats.norm.cdf(z_scores)
        
        return probabilities
    
    def analyze_calibration(self, n_bins=10):
        """
        NEW: Analyze model calibration - do predicted probabilities match reality?
        
        Args:
            n_bins: Number of probability bins
        
        Returns:
            DataFrame with calibration analysis
        """
        calibration_results = []
        
        for line in [5.5, 6.0, 6.5]:
            # Get predicted probabilities
            if self.prediction_type == 'poisson':
                pred_probs = np.array([
                    self.calculate_over_probability_poisson(line, i) 
                    for i in range(len(self.y_pred_total))
                ])
            else:
                pred_probs = self.calculate_over_probability_empirical(line)
            
            # Actual outcomes
            actual_over = (self.y_true_total > line).astype(float)
            
            # Bin the predictions
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(pred_probs, bins) - 1
            
            for i in range(n_bins):
                mask = bin_indices == i
                if mask.sum() > 0:
                    calibration_results.append({
                        'line': line,
                        'bin_start': bins[i],
                        'bin_end': bins[i+1],
                        'bin_center': (bins[i] + bins[i+1]) / 2,
                        'predicted_prob': pred_probs[mask].mean(),
                        'actual_freq': actual_over[mask].mean(),
                        'n_games': mask.sum(),
                        'calibration_error': pred_probs[mask].mean() - actual_over[mask].mean()
                    })
        
        return pd.DataFrame(calibration_results)
    
    def plot_enhanced_analysis(self, save_path=None):
        """
        ENHANCED: Create comprehensive visualization including residual analysis
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        # === ROW 1: Score Predictions ===
        # Home score predictions
        axes[0, 0].scatter(self.y_true_home, self.y_pred_home, alpha=0.5, s=10)
        axes[0, 0].plot([0, 8], [0, 8], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Home Score')
        axes[0, 0].set_ylabel('Predicted Home Score')
        axes[0, 0].set_title(f'Home Score Predictions\nMAE: {mean_absolute_error(self.y_true_home, self.y_pred_home):.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Away score predictions
        axes[0, 1].scatter(self.y_true_away, self.y_pred_away, alpha=0.5, s=10, color='orange')
        axes[0, 1].plot([0, 8], [0, 8], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Away Score')
        axes[0, 1].set_ylabel('Predicted Away Score')
        axes[0, 1].set_title(f'Away Score Predictions\nMAE: {mean_absolute_error(self.y_true_away, self.y_pred_away):.3f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total score predictions
        axes[0, 2].scatter(self.y_true_total, self.y_pred_total, alpha=0.5, s=10, color='green')
        axes[0, 2].plot([0, 15], [0, 15], 'r--', lw=2)
        axes[0, 2].set_xlabel('Actual Total Score')
        axes[0, 2].set_ylabel('Predicted Total Score')
        axes[0, 2].set_title(f'Total Score Predictions\nMAE: {mean_absolute_error(self.y_true_total, self.y_pred_total):.3f}')
        axes[0, 2].grid(True, alpha=0.3)
        
        # === ROW 2: Error Distributions ===
        axes[1, 0].hist(self.home_errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='red', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Home Error Distribution\nMean: {np.mean(self.home_errors):.3f}, Std: {self.home_error_std:.3f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(self.away_errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].axvline(0, color='red', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Away Error Distribution\nMean: {np.mean(self.away_errors):.3f}, Std: {self.away_error_std:.3f}')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].hist(self.total_errors, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1, 2].axvline(0, color='red', linestyle='--', lw=2)
        axes[1, 2].set_xlabel('Prediction Error')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title(f'Total Error Distribution\nMean: {np.mean(self.total_errors):.3f}, Std: {self.total_error_std:.3f}')
        axes[1, 2].grid(True, alpha=0.3)
        
        # === ROW 3: CRITICAL NEW PLOTS ===
        
        # RESIDUAL PLOT - Check for heteroscedasticity
        axes[2, 0].scatter(self.y_pred_total, self.total_errors, alpha=0.5, s=10, color='purple')
        axes[2, 0].axhline(0, color='red', linestyle='--', lw=2)
        axes[2, 0].set_xlabel('Predicted Total Score')
        axes[2, 0].set_ylabel('Prediction Error')
        axes[2, 0].set_title('Residual Plot\n(Should be random cloud)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Add trend line to check for patterns
        z = np.polyfit(self.y_pred_total, self.total_errors, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.y_pred_total.min(), self.y_pred_total.max(), 100)
        axes[2, 0].plot(x_trend, p(x_trend), 'g-', alpha=0.5, lw=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
        axes[2, 0].legend()
        
        # Q-Q PLOT - Check normality of errors
        stats.probplot(self.total_errors, dist="norm", plot=axes[2, 1])
        axes[2, 1].set_title('Q-Q Plot\n(Should be straight line for normal errors)')
        axes[2, 1].grid(True, alpha=0.3)
        
        # CALIBRATION PLOT
        calibration_df = self.analyze_calibration(n_bins=10)
        
        for line in [5.5, 6.0, 6.5]:
            line_data = calibration_df[calibration_df['line'] == line]
            axes[2, 2].scatter(line_data['bin_center'], line_data['actual_freq'], 
                             label=f'Line {line}', s=line_data['n_games']*2, alpha=0.7)
        
        axes[2, 2].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Calibration')
        axes[2, 2].set_xlabel('Predicted Probability')
        axes[2, 2].set_ylabel('Actual Frequency')
        axes[2, 2].set_title('Calibration Plot\n(Points should follow red line)')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def find_betting_edges_improved(self, min_edge=0.05, min_confidence=0.60, kelly_fraction=0.25):
        """
        IMPROVED: Find betting edges with empirical probabilities and Kelly sizing
        
        Args:
            min_edge: Minimum edge over implied probability
            min_confidence: Minimum confidence threshold
            kelly_fraction: Fraction of Kelly criterion to use (conservative)
        
        Returns:
            DataFrame with betting opportunities
        """
        edges = []
        
        common_lines = [5.5, 6.0, 6.5]
        
        for i in range(len(self.y_pred_total)):
            for line in common_lines:
                # Calculate probability with empirical std
                if self.prediction_type == 'poisson':
                    our_prob_over = self.calculate_over_probability_poisson(line, i)
                else:
                    our_prob_over = self.calculate_over_probability_empirical(line)[i]
                
                our_prob_under = 1 - our_prob_over
                
                # Standard -110 odds
                implied_prob = 0.524
                american_odds = -110
                decimal_odds = 1.91
                
                # Check for OVER edge
                if our_prob_over > implied_prob + min_edge and our_prob_over > min_confidence:
                    edge = our_prob_over - implied_prob
                    
                    # Kelly criterion for bet sizing
                    kelly_pct = (our_prob_over * decimal_odds - 1) / (decimal_odds - 1)
                    kelly_pct = max(0, kelly_pct) * kelly_fraction  # Conservative Kelly
                    
                    edges.append({
                        'game_idx': i,
                        'line': line,
                        'bet': 'OVER',
                        'our_probability': our_prob_over,
                        'implied_probability': implied_prob,
                        'edge': edge,
                        'kelly_pct': kelly_pct,
                        'predicted_total': self.y_pred_total[i],
                        'actual_total': self.y_true_total[i],
                        'hit': self.y_true_total[i] > line,
                        'confidence_level': 'HIGH' if our_prob_over > 0.65 else 'MEDIUM'
                    })
                
                # Check for UNDER edge
                elif our_prob_under > implied_prob + min_edge and our_prob_under > min_confidence:
                    edge = our_prob_under - implied_prob
                    
                    kelly_pct = (our_prob_under * decimal_odds - 1) / (decimal_odds - 1)
                    kelly_pct = max(0, kelly_pct) * kelly_fraction
                    
                    edges.append({
                        'game_idx': i,
                        'line': line,
                        'bet': 'UNDER',
                        'our_probability': our_prob_under,
                        'implied_probability': implied_prob,
                        'edge': edge,
                        'kelly_pct': kelly_pct,
                        'predicted_total': self.y_pred_total[i],
                        'actual_total': self.y_true_total[i],
                        'hit': self.y_true_total[i] < line,
                        'confidence_level': 'HIGH' if our_prob_under > 0.65 else 'MEDIUM'
                    })
        
        if edges:
            edge_df = pd.DataFrame(edges)
            edge_df['roi'] = (edge_df['hit'].astype(int) * 0.91 - 0.5) * 2
            return edge_df
        else:
            return pd.DataFrame()
    
    def generate_enhanced_report(self):
        """
        Generate comprehensive analysis report with all improvements
        """
        print("=" * 80)
        print("ENHANCED NHL SCORE PREDICTION & OVER/UNDER ANALYSIS")
        print("=" * 80)
        
        # 1. Score prediction metrics
        print("\n1. SCORE PREDICTION PERFORMANCE")
        print("-" * 40)
        
        metrics = {
            'home_mae': mean_absolute_error(self.y_true_home, self.y_pred_home),
            'away_mae': mean_absolute_error(self.y_true_away, self.y_pred_away),
            'total_mae': mean_absolute_error(self.y_true_total, self.y_pred_total),
            'home_rmse': np.sqrt(mean_squared_error(self.y_true_home, self.y_pred_home)),
            'away_rmse': np.sqrt(mean_squared_error(self.y_true_away, self.y_pred_away)),
            'total_rmse': np.sqrt(mean_squared_error(self.y_true_total, self.y_pred_total)),
        }
        
        print(f"Home Score  - MAE: {metrics['home_mae']:.3f}, RMSE: {metrics['home_rmse']:.3f}")
        print(f"Away Score  - MAE: {metrics['away_mae']:.3f}, RMSE: {metrics['away_rmse']:.3f}")
        print(f"Total Score - MAE: {metrics['total_mae']:.3f}, RMSE: {metrics['total_rmse']:.3f}")
        
        # 2. Error Analysis
        print("\n2. ERROR DISTRIBUTION ANALYSIS")
        print("-" * 40)
        print(f"Empirical Error Std Devs:")
        print(f"  Home:  {self.home_error_std:.3f} goals")
        print(f"  Away:  {self.away_error_std:.3f} goals")
        print(f"  Total: {self.total_error_std:.3f} goals")
        
        # Check for bias
        print(f"\nSystematic Bias:")
        print(f"  Home:  {np.mean(self.home_errors):+.3f} goals")
        print(f"  Away:  {np.mean(self.away_errors):+.3f} goals")
        print(f"  Total: {np.mean(self.total_errors):+.3f} goals")
        
        # 3. Calibration Analysis
        print("\n3. MODEL CALIBRATION")
        print("-" * 40)
        calibration_df = self.analyze_calibration()
        
        for line in [5.5, 6.0, 6.5]:
            line_cal = calibration_df[calibration_df['line'] == line]
            mae_cal = np.abs(line_cal['calibration_error']).mean()
            print(f"Line {line} - Mean Absolute Calibration Error: {mae_cal:.3f}")
        
        # 4. O/U Performance
        print("\n4. OVER/UNDER PERFORMANCE")
        print("-" * 40)
        
        for line in [5.5, 6.0, 6.5]:
            actual_over = (self.y_true_total > line).mean()
            
            if self.prediction_type == 'poisson':
                pred_probs = np.array([
                    self.calculate_over_probability_poisson(line, i) 
                    for i in range(len(self.y_pred_total))
                ])
            else:
                pred_probs = self.calculate_over_probability_empirical(line)
            
            pred_over = (pred_probs > 0.5).mean()
            
            # Accuracy when we predict
            confident_mask = (pred_probs > 0.55) | (pred_probs < 0.45)
            if confident_mask.sum() > 0:
                confident_acc = ((pred_probs[confident_mask] > 0.5) == 
                               (self.y_true_total[confident_mask] > line)).mean()
            else:
                confident_acc = 0
            
            print(f"\nLine {line}:")
            print(f"  Actual Over: {actual_over:.1%}")
            print(f"  Predicted Over: {pred_over:.1%}")
            print(f"  Confident Predictions: {confident_mask.mean():.1%} of games")
            print(f"  Accuracy on Confident: {confident_acc:.3f}")
        
        # 5. Betting Edges
        print("\n5. BETTING EDGE ANALYSIS")
        print("-" * 40)
        
        edges = self.find_betting_edges_improved()
        
        if not edges.empty:
            print(f"Found {len(edges)} betting opportunities")
            print(f"Average Edge: {edges['edge'].mean():.3f}")
            print(f"Win Rate: {edges['hit'].mean():.3f}")
            print(f"Expected ROI: {edges['roi'].mean():.3f}")
            
            # By confidence level
            for conf_level in ['HIGH', 'MEDIUM']:
                conf_edges = edges[edges['confidence_level'] == conf_level]
                if not conf_edges.empty:
                    print(f"\n{conf_level} Confidence:")
                    print(f"  Count: {len(conf_edges)}")
                    print(f"  Win Rate: {conf_edges['hit'].mean():.3f}")
                    print(f"  Avg Kelly %: {conf_edges['kelly_pct'].mean():.1%}")
        else:
            print("No significant edges found")
        
        print("\n" + "=" * 80)
        
        return metrics, calibration_df, edges


# Example usage function
def analyze_with_improvements(X_test, y_test_home, y_test_away, model_trainer, 
                              model_outputs_log=False):
    """
    Run improved analysis on model predictions
    
    Args:
        X_test: Test features
        y_test_home: True home scores
        y_test_away: True away scores
        model_trainer: Trained MultiTaskTrainer instance
        model_outputs_log: Set True if model outputs log-scores
    
    Returns:
        ImprovedScorePredictionAnalyzer instance
    """
    # Get predictions
    predictions = model_trainer.predict(X_test.values)
    
    # Create improved analyzer
    analyzer = ImprovedScorePredictionAnalyzer(
        y_test_home, y_test_away,
        predictions['home_score'], predictions['away_score'],
        prediction_type='raw',  # or 'poisson' if your model outputs lambda
        model_outputs_log=model_outputs_log
    )
    
    # Generate enhanced report
    metrics, calibration, edges = analyzer.generate_enhanced_report()
    
    # Create enhanced visualizations
    analyzer.plot_enhanced_analysis(save_path='enhanced_analysis.png')
    
    return analyzer


# Helper to convert existing model to Poisson
def add_poisson_output_layer(model):
    """
    Convert a regression model to output Poisson parameters
    
    This would modify your MultiTaskNHLModel to output lambda parameters
    instead of raw scores, enabling exact probability calculations
    """
    # This is pseudocode - actual implementation depends on your framework
    pass