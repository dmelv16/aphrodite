"""
Quantitative Betting Engine
Mathematical approach to optimal bet sizing without behavioral biases
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats, optimize
from scipy.interpolate import interp1d
import logging
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ThresholdData:
    """Empirical performance at each probability threshold"""
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    coverage_pct: float
    n_samples: int

@dataclass
class BetPosition:
    """Single betting position"""
    game_id: str
    team: str
    bet_type: str  # 'moneyline', 'spread', 'total_over', 'total_under'
    probability: float
    decimal_odds: float
    size: float
    timestamp: float

class QuantitativeBettingEngine:
    """
    Pure mathematical betting engine based on:
    1. Empirical accuracy interpolation
    2. Kelly Criterion with proper constraints
    3. Correlation-based portfolio optimization
    4. No behavioral adjustments or "intuition"
    """
    
    def __init__(self, 
                 bankroll: float = 1000,
                 max_portfolio_leverage: float = 0.25,
                 min_edge_threshold: float = 0.0,
                 calibration_file: str = 'src/models/saved/threshold_analysis.csv'):
        """
        Initialize Quantitative Betting Engine
        
        Args:
            bankroll: Starting bankroll
            max_portfolio_leverage: Maximum portion of bankroll at risk
            min_edge_threshold: Minimum edge required to place bet
            calibration_file: Path to threshold_analysis.csv from main.py
        """
        self.bankroll = bankroll
        self.max_portfolio_leverage = max_portfolio_leverage
        self.min_edge_threshold = min_edge_threshold
        
        # Load empirical threshold data from main.py calibration
        self.threshold_data = self._load_threshold_data(calibration_file)
        
        # Create interpolation functions for smooth accuracy estimates
        self.accuracy_interpolator = self._create_interpolators()
        
        # Portfolio tracking
        self.active_positions: List[BetPosition] = []
        
        # Correlation matrix (simplified - in production, calculate from historical data)
        self.correlation_matrix = self._initialize_correlation_matrix()
        
    def _load_threshold_data(self, calibration_file: str = None) -> pd.DataFrame:
        """Load empirical accuracy data from calibration file or use defaults"""
        
        # If calibration file provided and exists, try to load it
        if calibration_file:
            try:
                from pathlib import Path
                if Path(calibration_file).exists():
                    logger.info(f"Loading calibration data from {calibration_file}")
                    loaded_data = pd.read_csv(calibration_file)
                    if 'accuracy' in loaded_data.columns:
                        loaded_data = loaded_data.drop(columns=['accuracy'])
                    rename_map = {
                        'high_confidence_count': 'n_samples',
                        'coverage_pct': 'coverage',
                        'high_conf_accuracy': 'accuracy'  # <--- FIX: Uses the Home+Away High Conf Metric
                    }                    
                    loaded_data = loaded_data.rename(columns=rename_map)
                    
                    # If 'precision' was mapped to 'accuracy', we need to ensure we don't 
                    # have a collision if 'accuracy' meant something else.
                    # Ideally, drop the original 'accuracy' column first if it exists.
                    if 'accuracy' in loaded_data.columns and 'precision' in rename_map:
                         # Ensure we are using the mapped column, not the original 'accuracy'
                         pass
                    # Validate required columns
                    required_cols = {'threshold', 'accuracy', 'precision', 'n_samples'}
                    if required_cols.issubset(set(loaded_data.columns)):
                        # Rename n_samples to n if needed
                        if 'n_samples' in loaded_data.columns and 'n' not in loaded_data.columns:
                            loaded_data['n'] = loaded_data['n_samples']
                        
                        # Calculate coverage if not present
                        if 'coverage' not in loaded_data.columns:
                            total_samples = loaded_data['n'].sum()
                            loaded_data['coverage'] = (loaded_data['n'] / total_samples) * 100
                        
                        # Calculate standard error
                        loaded_data['std_error'] = np.sqrt(
                            loaded_data['accuracy'] * (1 - loaded_data['accuracy']) / loaded_data['n']
                        )
                        
                        logger.info(f"Successfully loaded {len(loaded_data)} calibration points")
                        return loaded_data
                    else:
                        logger.warning(f"Calibration file missing required columns. Using defaults.")
                else:
                    logger.warning(f"Calibration file not found: {calibration_file}. Using defaults.")
            except Exception as e:
                logger.warning(f"Error loading calibration file: {e}. Using defaults.")
        
        # Default calibration data (fallback)
        logger.info("Using default calibration data")
        data = pd.DataFrame([
            {'threshold': 0.50, 'accuracy': 0.5768, 'precision': 0.5871, 'coverage': 100.0, 'n': 3814},
            {'threshold': 0.55, 'accuracy': 0.6148, 'precision': 0.6216, 'coverage': 62.22, 'n': 2373},
            {'threshold': 0.60, 'accuracy': 0.6496, 'precision': 0.6546, 'coverage': 32.77, 'n': 1250},
            {'threshold': 0.65, 'accuracy': 0.6916, 'precision': 0.6975, 'coverage': 13.35, 'n': 509},
            {'threshold': 0.70, 'accuracy': 0.7037, 'precision': 0.7075, 'coverage': 4.25, 'n': 162},
            {'threshold': 0.75, 'accuracy': 0.7879, 'precision': 0.8387, 'coverage': 0.87, 'n': 33},
        ])
        
        # Calculate standard error for each threshold
        data['std_error'] = np.sqrt(data['accuracy'] * (1 - data['accuracy']) / data['n'])
        
        return data
    
    def _create_interpolators(self) -> Dict[str, interp1d]:
        """Create smooth interpolation functions for metrics"""
        df = self.threshold_data
        
        # Remove points with too few samples for reliable interpolation
        reliable_data = df[df['n'] >= 100].copy()
        
        interpolators = {
            'accuracy': interp1d(
                reliable_data['threshold'], 
                reliable_data['accuracy'],
                kind='cubic',
                bounds_error=False,
                fill_value=(reliable_data['accuracy'].iloc[0], reliable_data['accuracy'].iloc[-1])
            ),
            'std_error': interp1d(
                reliable_data['threshold'],
                reliable_data['std_error'],
                kind='linear',
                bounds_error=False,
                fill_value=(reliable_data['std_error'].iloc[0], reliable_data['std_error'].iloc[-1])
            ),
            'samples': interp1d(
                reliable_data['threshold'],
                reliable_data['n'],
                kind='linear',
                bounds_error=False,
                fill_value=(reliable_data['n'].iloc[0], 0)
            )
        }
        
        return interpolators
    
    def _initialize_correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Initialize correlation estimates between bet types
        In production, calculate from historical data
        """
        correlations = {
            # Same game correlations
            ('moneyline_same_game', 'total_over'): 0.3,    # Winner tends to score more
            ('moneyline_same_game', 'total_under'): -0.2,  # Defensive wins = lower scores
            ('moneyline_same_game', 'spread'): 0.8,        # ML and spread highly correlated
            
            # Cross-game correlations (same teams)
            ('same_team_different_game', 'any'): 0.1,      # Weak correlation across games
            
            # Default
            ('independent', 'independent'): 0.0
        }
        return correlations
    
    def get_empirical_accuracy(self, probability: float) -> Tuple[float, float]:
        """
        Get CONTINUOUS empirical accuracy via interpolation between DISCRETE buckets
        
        This is the key innovation: main.py provides discrete points (0.65, 0.70, etc.)
        but we interpolate smoothly between them for any probability value.
        
        Example:
            main.py says: 0.65 threshold → 69.16% actual accuracy
                         0.70 threshold → 70.37% actual accuracy
            
            If model predicts 0.67, we interpolate: ~69.64% accuracy
        
        Args:
            probability: Model's predicted probability (e.g., 0.67)
            
        Returns:
            (interpolated_accuracy, standard_error)
        """
        # Ensure probability is within interpolation range
        prob_clipped = np.clip(probability, 
                              self.threshold_data['threshold'].min(),
                              self.threshold_data['threshold'].max())
        
        # Use cubic interpolation for smooth accuracy curve
        accuracy = float(self.accuracy_interpolator['accuracy'](prob_clipped))
        std_error = float(self.accuracy_interpolator['std_error'](prob_clipped))
        
        # Log the interpolation for transparency
        if hasattr(self, '_debug_mode') and self._debug_mode:
            # Find surrounding buckets
            lower_bucket = self.threshold_data[self.threshold_data['threshold'] <= probability].iloc[-1] if len(self.threshold_data[self.threshold_data['threshold'] <= probability]) > 0 else None
            upper_bucket = self.threshold_data[self.threshold_data['threshold'] > probability].iloc[0] if len(self.threshold_data[self.threshold_data['threshold'] > probability]) > 0 else None
            
            if lower_bucket is not None and upper_bucket is not None:
                print(f"  Interpolating {probability:.3f} between:")
                print(f"    {lower_bucket['threshold']:.2f} → {lower_bucket['accuracy']:.1%}")
                print(f"    {upper_bucket['threshold']:.2f} → {upper_bucket['accuracy']:.1%}")
                print(f"  Result: {accuracy:.1%}")
        
        return accuracy, std_error
    
    def calculate_kelly_fraction(self,
                                win_probability: float,
                                decimal_odds: float,
                                confidence_interval_width: float = 1.96) -> float:
        """
        Calculate Kelly fraction with uncertainty adjustment
        
        Uses the Kelly formula: f = (bp - q) / b
        Adjusts for model uncertainty using confidence intervals
        """
        # Get empirical accuracy for this probability level
        empirical_accuracy, std_error = self.get_empirical_accuracy(win_probability)
        
        # Adjust probability based on empirical calibration
        # If model says 65% but empirical accuracy at 65% is 69%, use 69%
        calibration_ratio = empirical_accuracy / win_probability if win_probability > 0 else 1.0
        adjusted_probability = min(0.99, win_probability * calibration_ratio)
        
        # Calculate edge
        implied_probability = 1 / decimal_odds
        edge = adjusted_probability - implied_probability
        
        # No bet on negative or negligible edge
        if edge < self.min_edge_threshold:
            return 0.0
        
        # Standard Kelly calculation
        b = decimal_odds - 1  # Net odds
        p = adjusted_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Adjust for uncertainty (fractional Kelly based on confidence)
        # Higher uncertainty = lower fraction
        uncertainty_penalty = 1 / (1 + confidence_interval_width * std_error)
        
        # Apply standard fractional Kelly (25% is common in practice)
        conservative_fraction = 0.25
        
        return max(0, kelly_fraction * conservative_fraction * uncertainty_penalty)
    
    def calculate_portfolio_correlation(self, 
                                       new_position: BetPosition,
                                       existing_positions: List[BetPosition]) -> float:
        """
        Calculate correlation between new bet and existing portfolio
        
        Returns correlation coefficient between -1 and 1
        """
        if not existing_positions:
            return 0.0
        
        total_correlation = 0.0
        correlation_weights = 0.0
        
        for position in existing_positions:
            # Determine correlation type
            if position.game_id == new_position.game_id:
                if position.bet_type == new_position.bet_type:
                    corr = 1.0  # Duplicate bet
                elif 'moneyline' in position.bet_type and 'spread' in new_position.bet_type:
                    corr = 0.8  # ML and spread highly correlated
                elif 'over' in position.bet_type and 'moneyline' in new_position.bet_type:
                    corr = 0.3  # Winning team tends to score more
                else:
                    corr = 0.1  # Same game, different bet types
            elif position.team == new_position.team:
                corr = 0.1  # Same team, different games
            else:
                corr = 0.0  # Independent bets
            
            # Weight by position sizes
            weight = position.size * new_position.size
            total_correlation += corr * weight
            correlation_weights += weight
        
        if correlation_weights > 0:
            return total_correlation / correlation_weights
        return 0.0
    
    def optimize_bet_size(self,
                         base_kelly_size: float,
                         portfolio_correlation: float,
                         current_portfolio_exposure: float) -> float:
        """
        Optimize bet size considering portfolio constraints
        
        No behavioral adjustments - pure portfolio math
        """
        # Reduce size for highly correlated bets (avoid concentration risk)
        if portfolio_correlation > 0.5:
            correlation_adjustment = 1 / (1 + portfolio_correlation)
        else:
            correlation_adjustment = 1.0
        
        # Apply portfolio leverage constraint
        max_additional_exposure = (self.max_portfolio_leverage * self.bankroll) - current_portfolio_exposure
        
        # Final size is minimum of Kelly recommendation and portfolio constraint
        optimal_size = min(
            base_kelly_size * correlation_adjustment,
            max_additional_exposure,
            self.bankroll * 0.05  # Single bet limit (5% of bankroll)
        )
        
        return max(0, optimal_size)
    
    def calculate_expected_value(self,
                                probability: float,
                                decimal_odds: float,
                                bet_size: float) -> Dict[str, float]:
        """
        Calculate expected value and related metrics
        """
        # Get calibrated probability
        empirical_accuracy, std_error = self.get_empirical_accuracy(probability)
        calibration_ratio = empirical_accuracy / probability if probability > 0 else 1.0
        adjusted_probability = min(0.99, probability * calibration_ratio)
        
        # Expected value calculation
        win_payout = bet_size * decimal_odds
        expected_value = (adjusted_probability * win_payout) - ((1 - adjusted_probability) * bet_size)
        
        # Variance and standard deviation
        variance = adjusted_probability * (win_payout - expected_value) ** 2 + \
                  (1 - adjusted_probability) * (-bet_size - expected_value) ** 2
        std_dev = np.sqrt(variance)
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = expected_value / std_dev if std_dev > 0 else 0
        
        # Probability of profit
        profit_probability = adjusted_probability
        
        return {
            'expected_value': expected_value,
            'std_deviation': std_dev,
            'sharpe_ratio': sharpe_ratio,
            'profit_probability': profit_probability,
            'kelly_criterion': expected_value / variance if variance > 0 else 0
        }
    
    def make_betting_decision(self,
                             game_id: str,
                             team: str,
                             bet_type: str,
                             model_probability: float,
                             decimal_odds: float,
                             timestamp: Optional[float] = None) -> Dict:
        """
        Make optimal betting decision based on mathematical principles
        """
        # 1. Calculate Empirical Accuracy IMMEDIATELY (Fixes UnboundLocalError)
        # We calculate this first so it is available for all return paths (NO_BET or BET)
        empirical_accuracy, std_error = self.get_empirical_accuracy(model_probability)
        
        # Calculate base Kelly size
        kelly_fraction = self.calculate_kelly_fraction(model_probability, decimal_odds)
        base_kelly_size = kelly_fraction * self.bankroll
        
        # Check minimum edge
        implied_probability = 1 / decimal_odds
        edge = model_probability - implied_probability
        
        # 2. Check: Edge Threshold & Minimum Size
        if edge < self.min_edge_threshold or base_kelly_size < 10:
            return {
                'decision': 'NO_BET',
                'reason': 'INSUFFICIENT_EDGE' if edge < self.min_edge_threshold else 'SIZE_TOO_SMALL',
                'edge': edge,
                'kelly_fraction': kelly_fraction,
                'empirical_accuracy': empirical_accuracy, # Now safely defined
                'model_probability': model_probability,
                'implied_probability': implied_probability
            }
        
        # Create potential position
        new_position = BetPosition(
            game_id=game_id,
            team=team,
            bet_type=bet_type,
            probability=model_probability,
            decimal_odds=decimal_odds,
            size=base_kelly_size,
            timestamp=timestamp or 0
        )
        
        # Calculate portfolio metrics
        current_exposure = sum(p.size for p in self.active_positions)
        portfolio_correlation = self.calculate_portfolio_correlation(new_position, self.active_positions)
        
        # Optimize size with portfolio constraints
        optimal_size = self.optimize_bet_size(
            base_kelly_size,
            portfolio_correlation,
            current_exposure
        )
        
        # 3. Check: Portfolio Constraints
        if optimal_size < 10:  # Minimum bet threshold
            return {
                'decision': 'NO_BET',
                'reason': 'PORTFOLIO_CONSTRAINTS',
                'edge': edge,
                'kelly_fraction': kelly_fraction,
                'portfolio_correlation': portfolio_correlation,
                'empirical_accuracy': empirical_accuracy, # Added this (was missing!)
                'model_probability': model_probability
            }
        
        # Calculate expected value metrics
        ev_metrics = self.calculate_expected_value(model_probability, decimal_odds, optimal_size)
        
        return {
            'decision': 'BET',
            'bet_size': optimal_size,
            'bet_size_pct': optimal_size / self.bankroll * 100,
            'kelly_fraction': kelly_fraction,
            'edge': edge,
            'model_probability': model_probability,
            'empirical_accuracy': empirical_accuracy,
            'accuracy_std_error': std_error,
            'implied_probability': implied_probability,
            'decimal_odds': decimal_odds,
            'expected_value': ev_metrics['expected_value'],
            'std_deviation': ev_metrics['std_deviation'],
            'sharpe_ratio': ev_metrics['sharpe_ratio'],
            'profit_probability': ev_metrics['profit_probability'],
            'portfolio_correlation': portfolio_correlation,
            'current_portfolio_exposure': current_exposure + optimal_size,
            'game_id': game_id,
            'team': team,
            'bet_type': bet_type
        }
    
    def update_portfolio(self, position: BetPosition, result: str):
        """
        Update portfolio after bet resolution
        
        Args:
            position: The bet position
            result: 'WIN', 'LOSS', or 'PUSH'
        """
        # Remove from active positions
        self.active_positions = [p for p in self.active_positions 
                                if not (p.game_id == position.game_id and 
                                      p.bet_type == position.bet_type)]
        
        # Update bankroll
        if result == 'WIN':
            self.bankroll += position.size * (position.decimal_odds - 1)
        elif result == 'LOSS':
            self.bankroll -= position.size
        # PUSH returns the bet
    
    def get_portfolio_metrics(self) -> Dict:
        """
        Calculate current portfolio metrics
        """
        if not self.active_positions:
            return {
                'total_exposure': 0,
                'exposure_pct': 0,
                'position_count': 0,
                'avg_edge': 0,
                'portfolio_expected_value': 0,
                'portfolio_std_dev': 0
            }
        
        total_exposure = sum(p.size for p in self.active_positions)
        edges = [p.probability - (1/p.decimal_odds) for p in self.active_positions]
        
        # Portfolio EV (assuming independence for simplicity)
        portfolio_ev = sum(
            self.calculate_expected_value(p.probability, p.decimal_odds, p.size)['expected_value']
            for p in self.active_positions
        )
        
        # Portfolio standard deviation (simplified - assumes independence)
        portfolio_var = sum(
            self.calculate_expected_value(p.probability, p.decimal_odds, p.size)['std_deviation'] ** 2
            for p in self.active_positions
        )
        portfolio_std = np.sqrt(portfolio_var)
        
        return {
            'total_exposure': total_exposure,
            'exposure_pct': total_exposure / self.bankroll * 100,
            'position_count': len(self.active_positions),
            'avg_edge': np.mean(edges),
            'portfolio_expected_value': portfolio_ev,
            'portfolio_std_dev': portfolio_std,
            'portfolio_sharpe': portfolio_ev / portfolio_std if portfolio_std > 0 else 0
        }