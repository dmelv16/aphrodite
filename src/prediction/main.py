"""
NHL Today's Games Predictor with Betting Recommendations
Fetches today's games from NHL API, prepares features, and predicts using XGBoost with confidence thresholds
"""

import pandas as pd
import numpy as np
import joblib
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules
from data.loaders import NHLDataLoader
from features.temporal import TemporalFeatureEngine
from features.advanced import AdvancedFeatureEngine
from models.gradientBoosting import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TodaysGamesPredictor:
    """Predict today's NHL games with confidence-based betting recommendations"""
    
    def __init__(self, model_dir: str = 'models/saved', connection_string: str = None):
        self.model_dir = model_dir
        self.connection_string = connection_string
        self.model = None
        self.feature_columns = None
        self.data_loader = None
        self.temporal_engine = None
        self.advanced_engine = None
        
        # Betting thresholds (based on your analysis)
        self.thresholds = {
            'very_high': 0.65,  # 69.2% accuracy
            'high': 0.60,       # 65.0% accuracy
            'medium': 0.55,     # 61.5% accuracy
            'skip': 0.50        # 57.7% accuracy - don't bet
        }
        
        # Kelly fractions for each confidence level
        self.kelly_fractions = {
            'very_high': 1.0,   # Full Kelly
            'high': 0.5,        # Half Kelly
            'medium': 0.25,     # Quarter Kelly
            'skip': 0.0         # No bet
        }
    
    def load_model(self):
        """Load trained XGBoost model and feature columns"""
        logger.info(f"Loading model from {self.model_dir}")
        
        # Load feature columns
        self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.pkl")
        logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        
        # Load XGBoost model
        self.model = XGBoostModel(task='classification')
        self.model.load_model(f"{self.model_dir}/xgboost_outcome")
        logger.info("XGBoost model loaded successfully")
    
    def initialize_data_loader(self):
        """Initialize database connection and feature engines"""
        logger.info("Initializing data loader and feature engines...")
        
        self.data_loader = NHLDataLoader(self.connection_string)
        self.data_loader.connect()
        
        self.temporal_engine = TemporalFeatureEngine(windows=[5, 10, 20])
        self.advanced_engine = AdvancedFeatureEngine()
        
        logger.info("Data loader and feature engines ready")
    
    def fetch_todays_games(self) -> pd.DataFrame:
        """
        Fetch today's scheduled NHL games from NHL API
        
        Returns:
            DataFrame with today's games
        """
        today = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Fetching games from NHL API for {today}")
        
        # NHL API endpoint for today's schedule
        url = f"https://api-web.nhle.com/v1/schedule/{today}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            games_list = []
            
            # Parse the schedule
            if 'gameWeek' in data:
                for day in data['gameWeek']:
                    if 'games' in day:
                        for game in day['games']:
                            # Only include games that haven't started yet
                            game_state = game.get('gameState', '')
                            
                            if game_state in ['FUT', 'PRE']:  # Future or Pre-game
                                game_info = {
                                    'game_id': game['id'],
                                    'season': game.get('season', ''),
                                    'gameDate': day['date'],
                                    'gameState': game_state,
                                    'startTimeUTC': game.get('startTimeUTC', ''),
                                    'homeTeam_id': game['homeTeam']['id'],
                                    'homeTeam_abbrev': game['homeTeam']['abbrev'],
                                    'homeTeam_name': game['homeTeam'].get('placeName', {}).get('default', ''),
                                    'homeTeam_odds': game['homeTeam'].get('odds', []),
                                    'awayTeam_id': game['awayTeam']['id'],
                                    'awayTeam_abbrev': game['awayTeam']['abbrev'],
                                    'awayTeam_name': game['awayTeam'].get('placeName', {}).get('default', ''),
                                    'awayTeam_odds': game['awayTeam'].get('odds', []),
                                    'venue': game.get('venue', {}).get('default', '')
                                }
                                games_list.append(game_info)
            
            games_df = pd.DataFrame(games_list)
            
            if len(games_df) == 0:
                logger.warning(f"No upcoming games found for {today}")
            else:
                logger.info(f"Found {len(games_df)} upcoming games from NHL API")
                
                # Log the games with odds
                for _, game in games_df.iterrows():
                    home_odds = self._get_team_odds(game, 'home')[1]
                    away_odds = self._get_team_odds(game, 'away')[1]
                    logger.info(f"  {game['awayTeam_abbrev']} ({away_odds:+d}) @ {game['homeTeam_abbrev']} ({home_odds:+d}) at {game['startTimeUTC']}")
            
            return games_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch games from NHL API: {e}")
            
            # Fallback to database query
            logger.info("Attempting to fetch from database as fallback...")
            return self._fetch_todays_games_from_db()
        
        except Exception as e:
            logger.error(f"Error parsing NHL API response: {e}")
            return pd.DataFrame()
    
    def _fetch_todays_games_from_db(self) -> pd.DataFrame:
        """
        Fallback: Fetch today's games from database
        
        Returns:
            DataFrame with today's games
        """
        today = datetime.now().date()
        logger.info(f"Fetching games from database for {today}")
        
        query = f"""
        SELECT 
            game_id,
            season,
            gameDate,
            homeTeam_id,
            homeTeam_abbrev,
            homeTeam_placeName_default AS homeTeam_name,
            awayTeam_id,
            awayTeam_abbrev,
            awayTeam_placeName_default AS awayTeam_name,
            venue_default AS venue,
            startTimeUTC,
            gameState
        FROM schedule
        WHERE CAST(gameDate AS DATE) = '{today}'
            AND gameType = 2  -- Regular season games
            AND gameState IN ('FUT', 'PRE')  -- Future/Pre-game only
        ORDER BY startTimeUTC
        """
        
        try:
            games = pd.read_sql(query, self.data_loader.connection)
            
            if len(games) == 0:
                logger.warning(f"No games scheduled in database for {today}")
            else:
                logger.info(f"Found {len(games)} games in database")
            
            return games
        except Exception as e:
            logger.error(f"Failed to query database: {e}")
            return pd.DataFrame()
    
    def fetch_team_recent_data(self, team_id: int, as_of_date: datetime, n_games: int = 20) -> Dict:
        """
        Fetch recent game data for a team
        
        Returns:
            Dictionary with schedule and stats
        """
        # Get recent games
        query_schedule = f"""
        SELECT TOP {n_games}
            s.game_id,
            s.season,
            s.gameDate,
            CASE 
                WHEN s.homeTeam_id = {team_id} THEN s.homeTeam_score
                ELSE s.awayTeam_score
            END as team_score,
            CASE 
                WHEN s.homeTeam_id = {team_id} THEN s.awayTeam_score
                ELSE s.homeTeam_score
            END as opponent_score,
            CASE 
                WHEN s.homeTeam_id = {team_id} THEN 1
                ELSE 0
            END as is_home,
            CASE 
                WHEN s.homeTeam_id = {team_id} THEN s.awayTeam_id
                ELSE s.homeTeam_id
            END as opponent_id
        FROM schedule s
        WHERE (s.homeTeam_id = {team_id} OR s.awayTeam_id = {team_id})
            AND s.gameDate < '{as_of_date}'
            AND s.gameType = 2
            AND s.gameState = 'OFF'
        ORDER BY s.gameDate DESC
        """
        
        schedule = pd.read_sql(query_schedule, self.data_loader.connection)
        
        if len(schedule) == 0:
            logger.warning(f"No recent games found for team {team_id}")
            return None
        
        # Get team_game_xg stats for these games
        game_ids = schedule['game_id'].tolist()
        game_ids_str = ','.join(map(str, game_ids))
        
        query_stats = f"""
        SELECT *
        FROM team_game_xg
        WHERE team_id = {team_id}
            AND game_id IN ({game_ids_str})
        """
        
        stats = pd.read_sql(query_stats, self.data_loader.connection)
        
        # Merge
        team_data = schedule.merge(stats, on='game_id', how='left')
        
        return team_data
    
    def prepare_game_features(self, game_row: pd.Series) -> pd.DataFrame:
        """
        Prepare features for a single game
        
        Args:
            game_row: Row from todays_games DataFrame
            
        Returns:
            Feature DataFrame ready for prediction
        """
        home_id = game_row['homeTeam_id']
        away_id = game_row['awayTeam_id']
        game_date = pd.to_datetime(game_row['gameDate'])
        
        logger.info(f"Preparing features for {game_row['awayTeam_abbrev']} @ {game_row['homeTeam_abbrev']}")
        
        # Fetch recent data for both teams
        home_data = self.fetch_team_recent_data(home_id, game_date, n_games=20)
        away_data = self.fetch_team_recent_data(away_id, game_date, n_games=20)
        
        if home_data is None or away_data is None:
            logger.error(f"Could not fetch data for game {game_row['game_id']}")
            return None
        
        # Calculate temporal features
        home_features = self._calculate_team_features(home_data, is_home=True)
        away_features = self._calculate_team_features(away_data, is_home=False)
        
        # Combine into single row
        game_features = {}
        
        for key, value in home_features.items():
            game_features[f'home_{key}'] = value
        
        for key, value in away_features.items():
            game_features[f'away_{key}'] = value
        
        # Add matchup features
        game_features['day_of_week'] = game_date.dayofweek
        game_features['month'] = game_date.month
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([game_features])
        
        # Ensure all required features exist
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default value for missing features
        
        # Select only the features the model was trained on
        feature_df = feature_df[self.feature_columns]
        
        return feature_df
    
    def _calculate_team_features(self, team_data: pd.DataFrame, is_home: bool) -> Dict:
        """Calculate rolling features for a team"""
        features = {}
        
        if len(team_data) == 0:
            return self._get_default_features()
        
        # Sort by date (most recent first)
        team_data = team_data.sort_values('gameDate', ascending=False)
        
        # Rolling windows
        for window in [5, 10, 20]:
            n = min(window, len(team_data))
            recent = team_data.head(n)
            
            features[f'goals_rolling_{window}'] = recent['team_score'].mean()
            features[f'goals_against_rolling_{window}'] = recent['opponent_score'].mean()
            features[f'goal_diff_rolling_{window}'] = (recent['team_score'] - recent['opponent_score']).mean()
            
            # xG features if available
            if 'xGoalsFor' in recent.columns:
                features[f'xg_for_rolling_{window}'] = recent['xGoalsFor'].mean()
                features[f'xg_against_rolling_{window}'] = recent['xGoalsAgainst'].mean()
        
        # Recent form
        last_5 = team_data.head(5)
        features['wins_last_5'] = (last_5['team_score'] > last_5['opponent_score']).sum()
        features['win_pct_last_5'] = features['wins_last_5'] / 5.0
        
        # Rest days
        if len(team_data) >= 2:
            last_game = pd.to_datetime(team_data.iloc[0]['gameDate'])
            prev_game = pd.to_datetime(team_data.iloc[1]['gameDate'])
            features['rest_days'] = (last_game - prev_game).days
        else:
            features['rest_days'] = 2
        
        features['is_back_to_back'] = 1 if features['rest_days'] <= 1 else 0
        features['is_well_rested'] = 1 if features['rest_days'] >= 3 else 0
        
        # Home/Away splits
        if is_home:
            home_games = team_data[team_data['is_home'] == 1]
            if len(home_games) > 0:
                features['home_win_pct'] = (home_games['team_score'] > home_games['opponent_score']).mean()
                features['home_goals_avg'] = home_games['team_score'].mean()
            else:
                features['home_win_pct'] = 0.5
                features['home_goals_avg'] = 3.0
        else:
            away_games = team_data[team_data['is_home'] == 0]
            if len(away_games) > 0:
                features['away_win_pct'] = (away_games['team_score'] > away_games['opponent_score']).mean()
                features['away_goals_avg'] = away_games['team_score'].mean()
            else:
                features['away_win_pct'] = 0.5
                features['away_goals_avg'] = 3.0
        
        # Streaks
        features['current_streak'] = self._calculate_streak(team_data)
        
        return features
    
    def _calculate_streak(self, team_data: pd.DataFrame) -> int:
        """Calculate current win/loss streak"""
        if len(team_data) == 0:
            return 0
        
        results = (team_data['team_score'] > team_data['opponent_score']).astype(int)
        
        streak = 0
        current_result = results.iloc[0]
        
        for result in results:
            if result == current_result:
                streak += 1 if current_result == 1 else -1
            else:
                break
        
        return streak
    
    def _get_default_features(self) -> Dict:
        """Default features for teams with insufficient history"""
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
            'wins_last_5': 2,
            'win_pct_last_5': 0.4,
            'rest_days': 2,
            'is_back_to_back': 0,
            'is_well_rested': 0,
            'home_win_pct': 0.5,
            'away_win_pct': 0.5,
            'current_streak': 0
        }
    
    def classify_confidence(self, home_prob: float) -> Tuple[str, float, str]:
        """
        Classify prediction confidence
        
        Returns:
            (confidence_level, accuracy, recommended_action)
        """
        away_prob = 1 - home_prob
        max_prob = max(home_prob, away_prob)
        
        if max_prob >= self.thresholds['very_high']:
            return 'VERY HIGH', 0.692, 'BET'
        elif max_prob >= self.thresholds['high']:
            return 'HIGH', 0.650, 'BET'
        elif max_prob >= self.thresholds['medium']:
            return 'MEDIUM', 0.615, 'SMALL BET'
        else:
            return 'LOW', 0.577, 'SKIP'
    
    def calculate_kelly_bet(self, prob: float, odds: float, confidence_level: str, bankroll: float = 1000) -> float:
        """
        Calculate Kelly Criterion bet size
        
        Args:
            prob: Win probability (model prediction)
            odds: Decimal odds
            confidence_level: Confidence classification
            bankroll: Current bankroll
            
        Returns:
            Recommended bet size
        """
        # Get Kelly fraction for this confidence level
        kelly_frac = self.kelly_fractions.get(confidence_level.lower().replace(' ', '_'), 0.0)
        
        if kelly_frac == 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - p
        b = odds - 1
        q = 1 - prob
        
        kelly_full = (b * prob - q) / b
        kelly_full = max(0, kelly_full)  # No negative bets
        
        # Apply fractional Kelly
        kelly_bet = kelly_frac * kelly_full * bankroll
        
        # Cap at 5% of bankroll for safety
        kelly_bet = min(kelly_bet, bankroll * 0.05)
        
        return kelly_bet
    
    def predict_todays_games(self, bankroll: float = 1000, odds_type: str = 'american') -> pd.DataFrame:
        """
        Predict all of today's games with betting recommendations
        
        Args:
            bankroll: Starting bankroll for bet sizing
            odds_type: 'american' or 'decimal'
            
        Returns:
            DataFrame with predictions and betting recommendations
        """
        # Fetch today's games from NHL API
        games = self.fetch_todays_games()
        
        if len(games) == 0:
            logger.info("No games today or unable to fetch games")
            return pd.DataFrame()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PREDICTING {len(games)} GAMES")
        logger.info(f"{'='*80}\n")
        
        predictions = []
        
        for idx, game in games.iterrows():
            try:
                logger.info(f"[{idx+1}/{len(games)}] Processing: {game['awayTeam_abbrev']} @ {game['homeTeam_abbrev']}")
                
                # Prepare features
                features = self.prepare_game_features(game)
                
                if features is None:
                    logger.warning(f"  ⚠️  Skipping - could not prepare features")
                    continue
                
                # Get prediction
                probs = self.model.predict(features)[0]
                
                # Binary classification: 0 = Away Win, 1 = Home Win
                home_prob = probs if isinstance(probs, (int, float)) else probs
                away_prob = 1 - home_prob
                
                # Classify confidence
                confidence, accuracy, action = self.classify_confidence(home_prob)
                
                # Determine predicted winner
                if home_prob > away_prob:
                    predicted_winner = 'HOME'
                    predicted_team = game['homeTeam_abbrev']
                    win_prob = home_prob
                    team_type = 'home'
                else:
                    predicted_winner = 'AWAY'
                    predicted_team = game['awayTeam_abbrev']
                    win_prob = away_prob
                    team_type = 'away'
                
                logger.info(f"  ✓ Prediction: {predicted_team} ({win_prob:.1%}) - {confidence} confidence")
                
                # Get real market odds from game data
                decimal_odds, american_odds = self._get_team_odds(game, team_type)
                
                # Calculate Kelly bet
                kelly_bet = self.calculate_kelly_bet(
                    win_prob, 
                    decimal_odds, 
                    confidence, 
                    bankroll
                )
                
                # Calculate edge over market
                implied_prob = 1 / decimal_odds
                edge = win_prob - implied_prob
                
                prediction = {
                    'game_id': game['game_id'],
                    'game_time': game['startTimeUTC'],
                    'matchup': f"{game['awayTeam_abbrev']} @ {game['homeTeam_abbrev']}",
                    'home_team': game['homeTeam_abbrev'],
                    'away_team': game['awayTeam_abbrev'],
                    'venue': game.get('venue', ''),
                    'home_win_prob': home_prob,
                    'away_win_prob': away_prob,
                    'predicted_winner': predicted_winner,
                    'predicted_team': predicted_team,
                    'win_probability': win_prob,
                    'confidence_level': confidence,
                    'expected_accuracy': accuracy,
                    'action': action,
                    'decimal_odds': decimal_odds,
                    'american_odds': american_odds,
                    'implied_probability': implied_prob,
                    'edge': edge,
                    'kelly_bet_size': kelly_bet,
                    'kelly_pct_bankroll': (kelly_bet / bankroll) * 100
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"  ❌ Error predicting game: {e}")
                import traceback
                traceback.print_exc()
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETED: {len(predictions_df)} predictions generated")
        logger.info(f"{'='*80}\n")
        
        return predictions_df
    
    def print_betting_card(self, predictions_df: pd.DataFrame):
        """Print formatted betting card"""
        print("\n" + "="*100)
        print(f"🏒 NHL BETTING CARD - {datetime.now().strftime('%A, %B %d, %Y')}")
        print("="*100)
        
        if len(predictions_df) == 0:
            print("\n❌ No games today or no predictions available")
            return
        
        # Sort by confidence and then by win probability
        predictions_df = predictions_df.sort_values(
            ['confidence_level', 'win_probability'], 
            ascending=[True, False]
        )
        
        # Group by action
        bet_games = predictions_df[predictions_df['action'].isin(['BET', 'SMALL BET'])]
        skip_games = predictions_df[predictions_df['action'] == 'SKIP']
        
        print(f"\n📊 SUMMARY")
        print(f"  Total Games: {len(predictions_df)}")
        print(f"  Recommended Bets: {len(bet_games)}")
        print(f"  Skip: {len(skip_games)}")
        
        if len(bet_games) > 0:
            print("\n" + "-"*100)
            print("✅ RECOMMENDED BETS")
            print("-"*100)
            
            for idx, pred in bet_games.iterrows():
                print(f"\n🎯 {pred['matchup']} - {pred['game_time']}")
                print(f"   Venue: {pred['venue']}")
                print(f"   Pick: {pred['predicted_team']} ({pred['predicted_winner']})")
                print(f"   Win Probability: {pred['win_probability']:.1%}")
                print(f"   Market Odds: {pred['american_odds']:+d} ({pred['decimal_odds']:.2f}) → Implied: {pred['implied_probability']:.1%}")
                print(f"   🎲 Edge: {pred['edge']:+.1%} {'✅' if pred['edge'] > 0 else '❌'}")
                print(f"   Confidence: {pred['confidence_level']} (Expected Accuracy: {pred['expected_accuracy']:.1%})")
                print(f"   💰 Recommended Bet: ${pred['kelly_bet_size']:.2f} ({pred['kelly_pct_bankroll']:.2f}% of bankroll)")
                print(f"   {'-'*90}")
        
        if len(skip_games) > 0:
            print("\n" + "-"*100)
            print("⚠️  GAMES TO SKIP (Low Confidence)")
            print("-"*100)
            
            for idx, pred in skip_games.iterrows():
                print(f"\n   {pred['matchup']} - {pred['game_time']}")
                print(f"   Model Lean: {pred['predicted_team']} ({pred['win_probability']:.1%})")
                print(f"   ❌ Confidence too low - SKIP THIS GAME")
        
        print("\n" + "="*100)
        
        # Summary statistics
        if len(bet_games) > 0:
            total_bet = bet_games['kelly_bet_size'].sum()
            avg_prob = bet_games['win_probability'].mean()
            
            print("\n💵 BETTING SUMMARY:")
            print(f"  Total Recommended Stake: ${total_bet:.2f}")
            print(f"  Average Win Probability: {avg_prob:.1%}")
            print(f"  Number of Bets: {len(bet_games)}")
            
            # Count by confidence level
            confidence_counts = bet_games['confidence_level'].value_counts()
            print(f"\n  Breakdown by Confidence:")
            for conf_level in ['VERY HIGH', 'HIGH', 'MEDIUM']:
                count = confidence_counts.get(conf_level, 0)
                if count > 0:
                    print(f"    {conf_level}: {count} bet(s)")
            
            # Calculate expected value
            expected_wins = (bet_games['win_probability'] * bet_games['kelly_bet_size'] * bet_games['decimal_odds']).sum()
            expected_loss = ((1 - bet_games['win_probability']) * bet_games['kelly_bet_size']).sum()
            expected_profit = expected_wins - total_bet
            expected_roi = (expected_profit / total_bet) * 100 if total_bet > 0 else 0
            
            print(f"\n  📈 Expected Value:")
            print(f"    Expected Profit: ${expected_profit:+.2f}")
            print(f"    Expected ROI: {expected_roi:+.1f}%")
            print("\n" + "="*100)
        else:
            print("\n⚠️  No high-confidence bets today - sit this one out!")
            print("="*100)
    
    def run(self, bankroll: float = 1000):
        """Main execution method"""
        logger.info("Starting NHL Today's Games Predictor")
        
        # Load model
        self.load_model()
        
        # Initialize data loader
        self.initialize_data_loader()
        
        # Predict today's games
        predictions = self.predict_todays_games(bankroll=bankroll)
        
        # Print betting card
        self.print_betting_card(predictions)
        
        # Save predictions to CSV
        if len(predictions) > 0:
            output_file = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            predictions.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
        
        # Cleanup
        self.data_loader.disconnect()
        
        return predictions


def main():
    """Main execution"""
    
    # Database connection string
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=nhlDB;"
        "Trusted_Connection=yes;"
    )
    
    # Initialize predictor
    predictor = TodaysGamesPredictor(
        model_dir='models/saved',
        connection_string=connection_string
    )
    
    # Run predictions with $1000 bankroll
    predictions = predictor.run(bankroll=1000)


if __name__ == "__main__":
    main()