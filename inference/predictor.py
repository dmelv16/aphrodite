"""
Real-time prediction module for NHL games
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NHLPredictor:
    """Real-time NHL game predictions using trained models"""
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        self.models = {}
        self.feature_columns = None
        self.temporal_engine = None
        self.advanced_engine = None
        
    def load_models(self):
        """Load all trained models"""
        logger.info(f"Loading models from {self.model_dir}")
        
        # Load feature columns
        self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.pkl")
        
        # Load XGBoost model
        from models.gradientBoosting import XGBoostModel
        xgb_model = XGBoostModel(task='multiclass')
        xgb_model.load_model(f"{self.model_dir}/xgboost_outcome")
        self.models['xgboost_outcome'] = xgb_model
        
        # Load neural network
        import torch
        from models.neural import MultiTaskNHLModel, MultiTaskTrainer
        
        # Recreate model architecture
        input_dim = len(self.feature_columns)
        model = MultiTaskNHLModel(input_dim=input_dim)
        trainer = MultiTaskTrainer(model)
        trainer.load_checkpoint(f"{self.model_dir}/multitask_nn.pt")
        self.models['multitask_nn'] = trainer
        
        logger.info("All models loaded successfully")
    
    def load_feature_engines(self):
        """Initialize feature engineering engines"""
        from features.temporal import TemporalFeatureEngine
        from features.advanced import AdvancedFeatureEngine
        
        self.temporal_engine = TemporalFeatureEngine()
        self.advanced_engine = AdvancedFeatureEngine()
        
        logger.info("Feature engines initialized")
    
    def fetch_recent_games(self,
                          team_id: int,
                          n_games: int = 20,
                          data_loader=None) -> pd.DataFrame:
        """
        Fetch recent games for a team
        
        Args:
            team_id: NHL team ID
            n_games: Number of recent games
            data_loader: NHLDataLoader instance
        
        Returns:
            DataFrame with recent games
        """
        if data_loader is None:
            raise ValueError("Data loader required")
        
        as_of_date = datetime.now()
        recent_games = data_loader.get_recent_games(team_id, as_of_date, n_games)
        
        return recent_games
    
    def prepare_game_features(self,
                             home_team_id: int,
                             away_team_id: int,
                             game_date: datetime,
                             venue: str,
                             data_loader) -> pd.DataFrame:
        """
        Prepare features for a single upcoming game
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID  
            game_date: Game date
            venue: Venue name
            data_loader: NHLDataLoader instance
        
        Returns:
            Feature DataFrame ready for prediction
        """
        logger.info(f"Preparing features for {home_team_id} vs {away_team_id}")
        
        # Fetch recent games for both teams
        home_recent = self.fetch_recent_games(home_team_id, 20, data_loader)
        away_recent = self.fetch_recent_games(away_team_id, 20, data_loader)
        
        # Calculate temporal features
        home_features = self.calculate_team_features(home_recent, is_home=True)
        away_features = self.calculate_team_features(away_recent, is_home=False)
        
        # Combine features
        game_features = {}
        
        for key, value in home_features.items():
            game_features[f'home_{key}'] = value
        
        for key, value in away_features.items():
            game_features[f'away_{key}'] = value
        
        # Add game context
        game_features['venue'] = venue
        game_features['game_date'] = game_date
        game_features['day_of_week'] = game_date.weekday()
        game_features['month'] = game_date.month
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([game_features])
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default value
        
        # Select and order features
        feature_df = feature_df[self.feature_columns]
        
        return feature_df
    
    def calculate_team_features(self,
                               recent_games: pd.DataFrame,
                               is_home: bool) -> Dict[str, float]:
        """
        Calculate rolling features for a team
        
        Args:
            recent_games: Recent games DataFrame
            is_home: Whether team is playing at home
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        if len(recent_games) == 0:
            return self._get_default_features()
        
        # Rolling averages
        windows = [5, 10, 20]
        for window in windows:
            n = min(window, len(recent_games))
            recent_n = recent_games.head(n)
            
            features[f'goals_rolling_{window}'] = recent_n['team_score'].mean()
            features[f'goals_against_rolling_{window}'] = recent_n['opponent_score'].mean()
            features[f'goal_diff_rolling_{window}'] = (
                recent_n['team_score'] - recent_n['opponent_score']
            ).mean()
        
        # Form indicators
        last_5 = recent_games.head(5)
        features['wins_last_5'] = (last_5['team_score'] > last_5['opponent_score']).sum()
        features['points_last_5'] = (
            (last_5['team_score'] > last_5['opponent_score']).sum() * 2
        )
        
        # Rest days
        if len(recent_games) >= 2:
            features['rest_days'] = (recent_games.iloc[0]['gameDate'] - recent_games.iloc[1]['gameDate']).days
        else:
            features['rest_days'] = 3
        
        features['is_back_to_back'] = 1 if features['rest_days'] <= 1 else 0
        
        # Home/away splits
        if is_home:
            home_games = recent_games[recent_games['is_home'] == 1]
            if len(home_games) > 0:
                features['home_win_pct'] = (home_games['team_score'] > home_games['opponent_score']).mean()
            else:
                features['home_win_pct'] = 0.5
        else:
            away_games = recent_games[recent_games['is_home'] == 0]
            if len(away_games) > 0:
                features['away_win_pct'] = (away_games['team_score'] > away_games['opponent_score']).mean()
            else:
                features['away_win_pct'] = 0.5
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values for teams with no history"""
        return {
            'goals_rolling_5': 3.0,
            'goals_rolling_10': 3.0,
            'goals_rolling_20': 3.0,
            'goals_against_rolling_5': 3.0,
            'goals_against_rolling_10': 3.0,
            'goals_against_rolling_20': 3.0,
            'goal_diff_rolling_5': 0.0,
            'goal_diff_rolling_10': 0.0,
            'goal_diff_rolling_20': 0.0,
            'wins_last_5': 2.5,
            'points_last_5': 5.0,
            'rest_days': 2,
            'is_back_to_back': 0,
            'home_win_pct': 0.5,
            'away_win_pct': 0.5
        }
    
    def predict_game(self,
                    home_team_id: int,
                    away_team_id: int,
                    game_date: datetime,
                    venue: str,
                    data_loader,
                    return_details: bool = True) -> Dict:
        """
        Predict outcome of a single game
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Game date
            venue: Venue name
            data_loader: NHLDataLoader instance
            return_details: Return detailed predictions from all models
        
        Returns:
            Prediction dictionary
        """
        logger.info(f"Predicting game: {home_team_id} vs {away_team_id}")
        
        # Prepare features
        features = self.prepare_game_features(
            home_team_id, away_team_id, game_date, venue, data_loader
        )
        
        # Get predictions from all models
        predictions = {}
        
        # XGBoost outcome prediction
        xgb_proba = self.models['xgboost_outcome'].predict(features)
        predictions['xgboost'] = {
            'outcome_proba': xgb_proba[0].tolist(),
            'predicted_outcome': int(np.argmax(xgb_proba[0]))
        }
        
        # Neural network predictions
        nn_predictions = self.models['multitask_nn'].predict(features.values)
        predictions['neural_network'] = {
            'outcome_proba': nn_predictions['outcome'][0].tolist(),
            'home_score': float(nn_predictions['home_score'][0]),
            'away_score': float(nn_predictions['away_score'][0]),
            'predicted_outcome': int(np.argmax(nn_predictions['outcome'][0]))
        }
        
        # Ensemble prediction (average probabilities)
        ensemble_proba = (xgb_proba[0] + nn_predictions['outcome'][0]) / 2
        
        # Final prediction
        final_prediction = {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'game_date': game_date.strftime('%Y-%m-%d'),
            'venue': venue,
            'prediction': {
                'home_win_probability': float(ensemble_proba[0]),
                'away_win_probability': float(ensemble_proba[1]),
                'ot_probability': float(ensemble_proba[2]),
                'predicted_winner': 'HOME' if ensemble_proba[0] > ensemble_proba[1] else 'AWAY',
                'confidence': float(max(ensemble_proba)),
                'expected_home_score': predictions['neural_network']['home_score'],
                'expected_away_score': predictions['neural_network']['away_score']
            }
        }
        
        if return_details:
            final_prediction['model_details'] = predictions
        
        return final_prediction
    
    def predict_multiple_games(self,
                              games: List[Dict],
                              data_loader) -> pd.DataFrame:
        """
        Predict multiple games at once
        
        Args:
            games: List of game dictionaries with keys:
                   'home_team_id', 'away_team_id', 'game_date', 'venue'
            data_loader: NHLDataLoader instance
        
        Returns:
            DataFrame with all predictions
        """
        predictions = []
        
        for game in games:
            try:
                pred = self.predict_game(
                    game['home_team_id'],
                    game['away_team_id'],
                    game['game_date'],
                    game['venue'],
                    data_loader,
                    return_details=False
                )
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting game {game}: {e}")
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        return predictions_df
    
    def get_betting_recommendations(self,
                                   prediction: Dict,
                                   market_odds: Dict[str, float],
                                   min_edge: float = 0.05) -> List[Dict]:
        """
        Get betting recommendations based on edge over market odds
        
        Args:
            prediction: Prediction dictionary from predict_game()
            market_odds: Dictionary with 'home', 'away', 'ot' odds
            min_edge: Minimum edge required to recommend bet
        
        Returns:
            List of recommended bets
        """
        recommendations = []
        
        pred_proba = prediction['prediction']
        
        # Convert odds to implied probabilities
        implied_probs = {
            'home': 1 / market_odds.get('home', 2.0),
            'away': 1 / market_odds.get('away', 2.0),
            'ot': 1 / market_odds.get('ot', 3.0)
        }
        
        # Calculate edges
        edges = {
            'home': pred_proba['home_win_probability'] - implied_probs['home'],
            'away': pred_proba['away_win_probability'] - implied_probs['away'],
            'ot': pred_proba['ot_probability'] - implied_probs['ot']
        }
        
        # Find bets with positive edge
        for outcome, edge in edges.items():
            if edge >= min_edge:
                recommendations.append({
                    'outcome': outcome.upper(),
                    'edge': edge,
                    'predicted_probability': pred_proba[f'{outcome}_probability'] if outcome != 'ot' else pred_proba['ot_probability'],
                    'market_probability': implied_probs[outcome],
                    'market_odds': market_odds.get(outcome),
                    'kelly_fraction': edge / (market_odds.get(outcome, 2.0) - 1),
                    'confidence': 'HIGH' if edge > 0.10 else 'MEDIUM'
                })
        
        # Sort by edge
        recommendations.sort(key=lambda x: x['edge'], reverse=True)
        
        return recommendations


def example_usage():
    """Example of how to use the predictor"""
    from data.loaders import NHLDataLoader
    
    # Initialize predictor
    predictor = NHLPredictor(model_dir='models/saved')
    predictor.load_models()
    predictor.load_feature_engines()
    
    # Initialize data loader
    connection_string = "your_connection_string_here"
    data_loader = NHLDataLoader(connection_string)
    data_loader.connect()
    
    # Predict a single game
    prediction = predictor.predict_game(
        home_team_id=6,  # Boston Bruins
        away_team_id=10,  # Toronto Maple Leafs
        game_date=datetime(2025, 3, 15),
        venue='TD Garden',
        data_loader=data_loader
    )
    
    print(f"\nGame Prediction:")
    print(f"Home Win: {prediction['prediction']['home_win_probability']:.1%}")
    print(f"Away Win: {prediction['prediction']['away_win_probability']:.1%}")
    print(f"OT: {prediction['prediction']['ot_probability']:.1%}")
    print(f"Winner: {prediction['prediction']['predicted_winner']}")
    print(f"Expected Score: {prediction['prediction']['expected_home_score']:.1f} - {prediction['prediction']['expected_away_score']:.1f}")
    
    # Get betting recommendations
    market_odds = {'home': 1.95, 'away': 2.10, 'ot': 3.50}
    recommendations = predictor.get_betting_recommendations(prediction, market_odds)
    
    if recommendations:
        print(f"\nBetting Recommendations:")
        for rec in recommendations:
            print(f"  {rec['outcome']}: Edge={rec['edge']:.1%}, Confidence={rec['confidence']}")
    
    data_loader.disconnect()


if __name__ == "__main__":
    example_usage()