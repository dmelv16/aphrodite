"""
NHL Today's Games Predictor with Betting Recommendations
Fetches today's games from NHL API, prepares features using EXACT same pipeline as training
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
import pytz
warnings.filterwarnings('ignore')

# Import your custom modules
from src.data.loaders import NHLDataLoader
from src.features.temporal import TemporalFeatureEngine
from src.features.advanced import AdvancedFeatureEngine
from src.models.gradientBoosting import XGBoostModel
from src.prediction.validation import FeatureValidator
from src.prediction.bet import QuantitativeBettingEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TodaysGamesPredictor:
    """Predict today's NHL games with confidence-based betting recommendations"""
    
    def __init__(self, model_dir: str = 'src/models/saved', connection_string: str = None):
        self.model_dir = model_dir
        self.connection_string = connection_string
        self.model = None
        self.feature_columns = None
        self.data_loader = None
        self.temporal_engine = None
        self.advanced_engine = None

    def _classify_edge_dynamic(self, edge: float, all_edges: list = None) -> str:
        """
        Classify edge dynamically based on distribution
        
        Args:
            edge: The edge value to classify
            all_edges: List of all edges in current predictions for percentile calculation
        """
        if edge < 0:
            return "NEGATIVE"
        
        if all_edges and len(all_edges) >= 3:
            # Use percentiles from current batch
            import numpy as np
            positive_edges = [e for e in all_edges if e > 0]
            
            if not positive_edges:
                # All edges are negative or zero
                return "MINIMAL" if edge >= 0 else "NEGATIVE"
            
            # Calculate percentiles
            p25 = np.percentile(positive_edges, 25)
            p50 = np.percentile(positive_edges, 50)
            p75 = np.percentile(positive_edges, 75)
            p90 = np.percentile(positive_edges, 90)
            
            # Classify based on percentiles
            if edge >= p90:
                return "EXCEPTIONAL"
            elif edge >= p75:
                return "STRONG"
            elif edge >= p50:
                return "GOOD"
            elif edge >= p25:
                return "MODERATE"
            else:
                return "MARGINAL"
        else:
            # Use statistical thresholds when we don't have enough data
            # Based on typical betting market efficiency
            mean_edge = 0.02  # Typical average edge in efficient markets
            std_edge = 0.015  # Typical standard deviation
            
            z_score = (edge - mean_edge) / std_edge
            
            if z_score >= 2.0:  # 2+ std deviations above mean
                return "EXCEPTIONAL"
            elif z_score >= 1.0:  # 1-2 std deviations above mean
                return "STRONG"
            elif z_score >= 0:  # Above mean
                return "GOOD"
            elif z_score >= -0.5:  # Slightly below mean but positive
                return "MODERATE"
            else:
                return "MARGINAL"    

    def load_model(self):
        """Load trained XGBoost model and feature columns"""
        logger.info(f"Loading model from {self.model_dir}")
        
        # Load feature columns
        self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.pkl")
        logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        
        # Load XGBoost model
        self.model = XGBoostModel(task='classification')  # ✅ FIXED: Use 'classification' not 'binary'
        self.model.load_model(f"{self.model_dir}/xgboost_outcome")
        logger.info("XGBoost model loaded successfully")

    def _convert_to_mountain_time(self, utc_time_str: str) -> str:
        """Convert UTC time to Mountain Time"""
        try:
            utc_time = datetime.strptime(utc_time_str.replace('Z', '+00:00'), '%Y-%m-%dT%H:%M:%S%z')
            mountain_tz = pytz.timezone('America/Denver')
            mountain_time = utc_time.astimezone(mountain_tz)
            return mountain_time.strftime('%I:%M %p MT')
        except:
            return utc_time_str
    
    def _get_today_mt(self) -> str:
        """Get today's date in Mountain Time"""
        mountain_tz = pytz.timezone('America/Denver')
        today_mt = datetime.now(mountain_tz).date()
        return today_mt.strftime('%Y-%m-%d')
        # return '2025-11-22'
        
    def run_comprehensive_diagnostics(self, features: pd.DataFrame, game_info: str):
        """
        Run comprehensive diagnostics on prepared features
        
        Add this to TodaysGamesPredictor class
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPREHENSIVE DIAGNOSTICS: {game_info}")
        logger.info(f"{'='*80}")
        
        # Basic stats
        logger.info(f"\nFeature Statistics:")
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Memory usage: {features.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Check for common feature patterns
        feature_groups = {
            'rolling': [f for f in features.columns if 'rolling' in f.lower()],
            'ema': [f for f in features.columns if 'ema' in f.lower()],
            'trend': [f for f in features.columns if 'trend' in f.lower()],
            'home': [f for f in features.columns if f.startswith('home_')],
            'away': [f for f in features.columns if f.startswith('away_')],
        }
        
        logger.info(f"\nFeature Groups:")
        for group_name, group_features in feature_groups.items():
            logger.info(f"  {group_name}: {len(group_features)} features")
        
        # Value distribution check
        logger.info(f"\nValue Distribution Check:")
        zero_count = (features == 0).sum().sum()
        total_values = features.shape[0] * features.shape[1]
        zero_pct = (zero_count / total_values) * 100
        
        logger.info(f"  Zero values: {zero_count} / {total_values} ({zero_pct:.1f}%)")
        
        if zero_pct > 50:
            logger.warning(f"  ⚠️  WARNING: >50% of values are zero - check data availability")
        
        # Feature correlation check (basic)
        if len(features.columns) > 1:
            # Check for perfect correlations (potential duplicates)
            corr_matrix = features.T.corr()
            perfect_corr = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.99:
                        perfect_corr.append(
                            f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}"
                        )
            
            if perfect_corr:
                logger.warning(f"\nHighly correlated features found ({len(perfect_corr)}):")
                for pair in perfect_corr[:5]:
                    logger.warning(f"  {pair}")


    def predict_with_validation(self, game_row: pd.Series) -> Dict:
        """
        Predict with comprehensive validation
        
        Replace the prediction logic in predict_todays_games with this
        
        Returns:
            Dict with prediction results and validation status
        """
        game_info = f"{game_row['awayTeam_abbrev']} @ {game_row['homeTeam_abbrev']}"
        
        # Prepare features
        features = self.prepare_game_features(game_row)
        
        if features is None:
            return {
                'valid': False,
                'error': 'Failed to prepare features',
                'game_info': game_info
            }
        
        # Make prediction
        try:
            probs = self.model.predict(features)[0]
            home_prob = probs if isinstance(probs, (int, float)) else probs
            away_prob = 1 - home_prob
            
            return {
                'valid': True,
                'home_prob': home_prob,
                'away_prob': away_prob,
                'features': features,
                'game_info': game_info
            }
        
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'valid': False,
                'error': f'Prediction failed: {str(e)}',
                'game_info': game_info
            }
        
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
        Fetch today's scheduled NHL games from NHL API (Mountain Time)
        
        Returns:
            DataFrame with today's games
        """
        # Get today's date in Mountain Time
        today_mt = self._get_today_mt()
        logger.info(f"Fetching games from NHL API for {today_mt} (Mountain Time)")
        
        # NHL API endpoint for today's schedule
        url = f"https://api-web.nhle.com/v1/schedule/{today_mt}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            games_list = []
            
            # Parse the schedule
            if 'gameWeek' in data:
                for day in data['gameWeek']:
                    # ✅ CRITICAL FIX: Only process games from TODAY
                    if day['date'] != today_mt:
                        continue
                        
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
                logger.warning(f"No upcoming games found for {today_mt}")
            else:
                logger.info(f"Found {len(games_df)} upcoming games for {today_mt}")
                
                # Log the games
                for _, game in games_df.iterrows():
                    try:
                        home_odds = self._get_team_odds(game, 'home')[1]
                        away_odds = self._get_team_odds(game, 'away')[1]
                        mt_time = self._convert_to_mountain_time(game['startTimeUTC'])
                        logger.info(f"  {game['awayTeam_abbrev']} ({away_odds:+d}) @ {game['homeTeam_abbrev']} ({home_odds:+d}) at {mt_time}")
                    except Exception as e:
                        mt_time = self._convert_to_mountain_time(game['startTimeUTC'])
                        logger.info(f"  {game['awayTeam_abbrev']} @ {game['homeTeam_abbrev']} at {mt_time}")
            
            return games_df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch games from NHL API: {e}")
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error parsing NHL API response: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _get_team_odds(self, game_row: pd.Series, team_type: str) -> Tuple[float, int]:
        """Extract betting odds for a team from game data"""
        team_key = f'{team_type}Team_odds'
        
        # Check if odds key exists in the game row
        if team_key not in game_row.index:
            logger.warning(f"No odds available for {team_type} team in this game, using default -110")
            return 1.91, -110
        
        odds_data = game_row[team_key]
        
        # Check if odds data is missing or empty
        if odds_data is None or (isinstance(odds_data, float) and pd.isna(odds_data)):
            logger.warning(f"No odds available for {team_type} team, using default -110")
            return 1.91, -110
        
        # If it's an empty list
        if isinstance(odds_data, list) and len(odds_data) == 0:
            logger.warning(f"No odds providers available for {team_type} team, using default -110")
            return 1.91, -110
        
        # If it's a string representation of a list, try to parse it
        if isinstance(odds_data, str):
            try:
                import json
                odds_data = json.loads(odds_data)
            except:
                logger.warning(f"Could not parse odds string for {team_type} team, using default -110")
                return 1.91, -110
        
        # Process list of odds from different providers
        if isinstance(odds_data, list) and len(odds_data) > 0:
            # Try each provider until we find valid odds
            for odds_item in odds_data:
                if isinstance(odds_item, dict) and 'value' in odds_item:
                    odds_value = odds_item['value']
                    
                    try:
                        # Handle decimal odds (e.g., "1.62", "2.23")
                        if isinstance(odds_value, str) and '.' in odds_value and not odds_value.startswith(('+', '-')):
                            decimal_odds = float(odds_value)
                            american_odds = self._decimal_to_american(decimal_odds)
                            provider_id = odds_item.get('providerId', 'unknown')
                            logger.info(f"  Using odds from provider {provider_id}: {american_odds:+d} (decimal: {decimal_odds:.2f})")
                            return decimal_odds, american_odds
                        
                        # Handle American odds (e.g., "+130", "-156")
                        elif isinstance(odds_value, str) and (odds_value.startswith('+') or odds_value.startswith('-')):
                            american_odds = int(odds_value)
                            decimal_odds = self._american_to_decimal(american_odds)
                            provider_id = odds_item.get('providerId', 'unknown')
                            logger.info(f"  Using odds from provider {provider_id}: {american_odds:+d} (decimal: {decimal_odds:.2f})")
                            return decimal_odds, american_odds
                        
                        # Handle numeric decimal odds
                        elif isinstance(odds_value, (int, float)):
                            decimal_odds = float(odds_value)
                            american_odds = self._decimal_to_american(decimal_odds)
                            provider_id = odds_item.get('providerId', 'unknown')
                            logger.info(f"  Using odds from provider {provider_id}: {american_odds:+d} (decimal: {decimal_odds:.2f})")
                            return decimal_odds, american_odds
                    
                    except (ValueError, ZeroDivisionError) as e:
                        logger.warning(f"  Could not parse odds value '{odds_value}': {e}")
                        continue
        
        # If we get here, no valid odds were found
        logger.warning(f"No valid odds found for {team_type} team after checking all providers, using default -110")
        return 1.91, -110

    def _american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return int(round((decimal_odds - 1) * 100))
        else:
            return int(round(-100 / (decimal_odds - 1)))

    def prepare_game_features(self, game_row: pd.Series) -> pd.DataFrame:
        """
        Prepare features for a single game by grafting historical stats onto today's matchup.
        """
        # 1. Setup Metadata from Today's Game
        current_home_id = int(game_row['homeTeam_id'])
        current_away_id = int(game_row['awayTeam_id'])
        game_date = pd.to_datetime(game_row['gameDate'])
        
        logger.info(f"Preparing features for {game_row['awayTeam_abbrev']} @ {game_row['homeTeam_abbrev']}")

        # 2. Fetch Historical Data
        historical_data = self._fetch_complete_historical_data(
            current_home_id, current_away_id, game_date, lookback_games=100
        )
        
        if historical_data is None:
            return None
            
        schedule = historical_data['schedule']
        shot_xg = historical_data['shot_xg']
        team_game_xg = historical_data['team_game_xg']
        events = historical_data['events']

        # 3. STRICT Type Enforcement & Deduplication
        try:
            schedule = schedule.drop_duplicates(subset=['game_id'])
            team_game_xg = team_game_xg.drop_duplicates(subset=['game_id', 'team_id'])
            
            for df in [schedule, team_game_xg, shot_xg, events]:
                if 'game_id' in df.columns: df['game_id'] = df['game_id'].astype(int)
                if 'team_id' in df.columns: df['team_id'] = df['team_id'].astype(int)
                if 'homeTeam_id' in df.columns: df['homeTeam_id'] = df['homeTeam_id'].astype(int)
                if 'awayTeam_id' in df.columns: df['awayTeam_id'] = df['awayTeam_id'].astype(int)
                if 'event_owner_team_id' in df.columns: 
                    df['event_owner_team_id'] = df['event_owner_team_id'].fillna(-1).astype(int)

        except Exception as e:
            logger.error(f"Data type normalization failed: {e}")
            return None

        # 4. Prepare Team Stats
        team_stats = []
        for _, game in schedule.iterrows():
            # Home logic
            h_stats = team_game_xg[
                (team_game_xg['game_id'] == game['game_id']) & 
                (team_game_xg['team_id'] == game['homeTeam_id'])
            ]
            if not h_stats.empty:
                s = h_stats.iloc[0].to_dict()
                s.update({
                    'is_home': 1,
                    'won': 1 if game['homeTeam_score'] > game['awayTeam_score'] else 0,
                    'goal_differential': game['homeTeam_score'] - game['awayTeam_score'],
                    'gameDate': game['gameDate']
                })
                s['points'] = 2 if s['won'] else (1 if self._detect_overtime(game.get('periodDescriptor_json')) else 0)
                team_stats.append(s)

            # Away logic
            a_stats = team_game_xg[
                (team_game_xg['game_id'] == game['game_id']) & 
                (team_game_xg['team_id'] == game['awayTeam_id'])
            ]
            if not a_stats.empty:
                s = a_stats.iloc[0].to_dict()
                s.update({
                    'is_home': 0,
                    'won': 1 if game['awayTeam_score'] > game['homeTeam_score'] else 0,
                    'goal_differential': game['awayTeam_score'] - game['homeTeam_score'],
                    'gameDate': game['gameDate']
                })
                s['points'] = 2 if s['won'] else (1 if self._detect_overtime(game.get('periodDescriptor_json')) else 0)
                team_stats.append(s)
        
        team_stats_df = pd.DataFrame(team_stats)

        numeric_cols = [
            'goals_for', 'goals_against', 'xG_for', 'xG_against', 
            'shots_for', 'shots_against', 'shooting_percentage', 'save_percentage',
            'won', 'points', 'goal_differential'
        ]
        
        for col in numeric_cols:
            if col in team_stats_df.columns:
                team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce').fillna(0)
        
        # 2. Sort strictly by Date/Team before calculating rolling features
        # (Temporal engine does this, but safe to ensure here for the Ghost Row insertion later)
        team_stats_df = team_stats_df.sort_values(['team_id', 'gameDate'])

        # 1. Calculate Season
        current_season = int(str(game_date.year) + str(game_date.year + 1)) \
                         if game_date.month > 8 else int(str(game_date.year - 1) + str(game_date.year))

        # 2. Inject Ghost Rows into TEAM_STATS (For Temporal Engine)
        # ---------------------------------------------------------
        home_ghost = {
            'game_id': 0, 'team_id': current_home_id, 'gameDate': game_date, 
            'is_home': 1, 'season': current_season
        }
        away_ghost = {
            'game_id': 0, 'team_id': current_away_id, 'gameDate': game_date, 
            'is_home': 0, 'season': current_season
        }
        
        # Fill numeric cols with NaN (prevents skewing mean, allows row to exist)
        for col in numeric_cols:
            home_ghost[col] = np.nan
            away_ghost[col] = np.nan

        team_stats_df = pd.concat([
            team_stats_df, 
            pd.DataFrame([home_ghost]), 
            pd.DataFrame([away_ghost])
        ], ignore_index=True)
        
        # 3. Inject Ghost Row into SCHEDULE (For Merging)
        # ---------------------------------------------------------
        ghost_schedule = pd.DataFrame([{
            'game_id': 0, 'gameDate': game_date, 
            'homeTeam_id': current_home_id, 'awayTeam_id': current_away_id,
            'season': current_season,
            'homeTeam_score': 0, 'awayTeam_score': 0 # Placeholders
        }])
        schedule = pd.concat([schedule, ghost_schedule], ignore_index=True)

        # 4. Inject Ghost Rows into TEAM_GAME_XG (Base for Advanced Engine)
        # ---------------------------------------------------------
        # This is critical because Advanced Engine merges onto this
        xg_ghost_h = {'game_id': 0, 'team_id': current_home_id, 'gameDate': game_date}
        xg_ghost_a = {'game_id': 0, 'team_id': current_away_id, 'gameDate': game_date}
        
        # Add required columns with 0 or NaN
        xg_req_cols = ['goals_for', 'goals_against', 'xG_for', 'xG_against', 
                       'shooting_percentage', 'save_percentage', 'is_home']
        for col in xg_req_cols:
            xg_ghost_h[col] = 0.0
            xg_ghost_a[col] = 0.0

        team_game_xg = pd.concat([
            team_game_xg, 
            pd.DataFrame([xg_ghost_h]), 
            pd.DataFrame([xg_ghost_a])
        ], ignore_index=True)

        # 5. Inject Ghost Rows into SHOT_XG (For Shot Quality Features)
        # ---------------------------------------------------------
        # We need at least one row per team for Game 0 so groupby() doesn't drop it.
        # The values don't matter because they are "Today's Stats" which get shifted out.
        # We just need the row to exist to catch the incoming shift from yesterday.
        shot_ghost_h = {
            'game_id': 0, 'event_owner_team_id': current_home_id, 
            'xG': 0.0, 'is_slot': 0, 'is_rebound': 0, 'is_rush': 0, 
            'distance': 0.0, 'angle': 0.0,
            'is_even_strength': 0, 'is_powerplay': 0, 'is_shorthanded': 0
        }
        shot_ghost_a = shot_ghost_h.copy()
        shot_ghost_a['event_owner_team_id'] = current_away_id
        
        shot_xg = pd.concat([
            shot_xg,
            pd.DataFrame([shot_ghost_h]),
            pd.DataFrame([shot_ghost_a])
        ], ignore_index=True)

        # 6. Inject Ghost Rows into EVENTS (For Event Features)
        # ---------------------------------------------------------
        event_ghost_h = {
            'game_id': 0, 'event_owner_team_id': current_home_id,
            'type_code': 0, 'event_id': 0, 'x_coord': 0, 'y_coord': 0,
            'penalty_duration': 0, 'zone_code': 'N'
        }
        event_ghost_a = event_ghost_h.copy()
        event_ghost_a['event_owner_team_id'] = current_away_id
        
        events = pd.concat([
            events,
            pd.DataFrame([event_ghost_h]),
            pd.DataFrame([event_ghost_a])
        ], ignore_index=True)

        # 5. Feature Engineering
        logger.info("Generating temporal features...")
        temporal_features = self.temporal_engine.generate_all_temporal_features(
            schedule, team_stats_df
        )
        
        logger.info("Generating advanced features...")
        advanced_features = self.advanced_engine.generate_all_advanced_features(
            schedule, shot_xg, team_game_xg, events
        )

        # 6. Consolidate Historical Features
        historical_master = schedule[['game_id', 'gameDate', 'homeTeam_id', 'awayTeam_id']].copy()
        
        # ✅ FIX: temporal_features and advanced_features are already game-level with home_/away_ prefixes
        temporal_features = temporal_features.drop_duplicates(subset=['game_id'])
        advanced_features = advanced_features.drop_duplicates(subset=['game_id'])

        # Simple merge - no splitting needed!
        historical_master = historical_master.merge(temporal_features, on='game_id', how='left')
        historical_master = historical_master.merge(advanced_features, on='game_id', how='left')
        
        historical_master = historical_master.sort_values('gameDate')
        logger.info(f"  Shape before cleanup: {historical_master.shape}")

        historical_master = consolidate_duplicate_features(historical_master)
        
        logger.info(f"  Shape after cleanup: {historical_master.shape}")
        
        def get_team_features(target_team_id, prefix_for_today):
            """Extract features specifically from the Ghost Row (game_id 0)"""
            
            # Look specifically for the row we injected (game_id=0)
            # This row contains the rolling stats ENTETING today's game
            team_games = historical_master[
                (historical_master['game_id'] == 0) & 
                ((historical_master['homeTeam_id'] == target_team_id) | 
                 (historical_master['awayTeam_id'] == target_team_id))
            ]
            
            if team_games.empty:
                # Fallback (Safety Net)
                logger.warning(f"Ghost row missing for {target_team_id}, falling back to last game")
                team_games = historical_master[
                    (historical_master['homeTeam_id'] == target_team_id) | 
                    (historical_master['awayTeam_id'] == target_team_id)
                ]
                if team_games.empty: return None
                last_game = team_games.iloc[-1]
            else:
                last_game = team_games.iloc[0]

            extracted = {}
            was_home_last_time = (last_game['homeTeam_id'] == target_team_id)
            
            # Get all feature columns (excluding metadata)
            feature_cols = [c for c in last_game.index 
                        if c not in ['game_id', 'gameDate', 'homeTeam_id', 'awayTeam_id']]
            
            for col in feature_cols:
                value = last_game[col]
                
                # Extract features based on role in last game
                if was_home_last_time:
                    if col.startswith('home_'):
                        # This team was home, extract their home_ features
                        clean_name = col.replace('home_', '')
                        extracted[f"{prefix_for_today}{clean_name}"] = value
                    elif col.startswith('away_'):
                        # Skip opponent's features
                        continue
                    else:
                        # Non-prefixed features (shared between both teams) - NEED THESE!
                        extracted[f"{prefix_for_today}{col}"] = value
                else:
                    if col.startswith('away_'):
                        # This team was away, extract their away_ features
                        clean_name = col.replace('away_', '')
                        extracted[f"{prefix_for_today}{clean_name}"] = value
                    elif col.startswith('home_'):
                        # Skip opponent's features
                        continue
                    else:
                        # Non-prefixed features (shared between both teams) - NEED THESE!
                        extracted[f"{prefix_for_today}{col}"] = value
            
            return extracted

        home_features_dict = get_team_features(current_home_id, 'home_')
        away_features_dict = get_team_features(current_away_id, 'away_')

        if not home_features_dict or not away_features_dict:
            return None

        # 8. Construct Final Prediction Row
        prediction_data = {
            'game_id': 0,
            'gameDate': game_date,
            'homeTeam_id': current_home_id,
            'awayTeam_id': current_away_id,
            **home_features_dict,
            **away_features_dict
        }
        
        feature_df = pd.DataFrame([prediction_data])

        # 9. Final Polish & Type Safety
        # Fill missing columns with 0
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
                
        feature_df = feature_df[self.feature_columns]
        
        # ==================================================================
        # ✅ CRITICAL FIX: Handle Categorical "Season Phase" columns manually
        # ==================================================================
        phase_mapping = {'early': 0, 'mid': 1, 'late': 2, 'playoff_push': 3}
        
        for col in feature_df.columns:
            # Check if this is one of the phase columns
            if 'season_phase' in col:
                logger.info(f"  Mapping season phase column: {col}")
                # If it's a string/object, map it
                if feature_df[col].dtype == 'object':
                    feature_df[col] = feature_df[col].map(phase_mapping).fillna(0)
                # If it's categorical, use codes
                elif feature_df[col].dtype.name == 'category':
                    feature_df[col] = feature_df[col].cat.codes

            # Handle other categorical columns
            elif feature_df[col].dtype.name == 'category':
                feature_df[col] = feature_df[col].cat.codes
                
        # Final numeric conversion
        feature_df = feature_df.fillna(0)
        try:
            feature_df = feature_df.astype(float)
        except ValueError as e:
            logger.error(f"Final float conversion failed: {e}")
            # Last ditch effort: force coerce everything
            for col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
        
        logger.info(f"Features prepared successfully: {feature_df.shape}")
        return feature_df

    def _fetch_complete_historical_data(self, 
                                        home_id: int, 
                                        away_id: int, 
                                        as_of_date: datetime,
                                        lookback_games: int = 100) -> Dict:
        """
        Fetch historical data using NHLDataLoader methods to avoid code duplication.
        """
        logger.info(f"  Fetching historical data (up to {as_of_date.date()})")
        
        try:
            # ============================================================
            # 1. Fetch RELEVANT Schedule (Custom query needed for efficiency)
            # ============================================================
            # We still use a custom query here because loaders.py doesn't have 
            # a "Last N games for Team A OR Team B" method.
            query_schedule = f"""
            SELECT TOP {lookback_games * 2}
                s.game_id,
                s.season,
                s.gameDate,
                s.homeTeam_id,
                s.awayTeam_id,
                s.homeTeam_score,
                s.awayTeam_score,
                s.periodDescriptor_json
            FROM [nhlDB].[schedule].[schedule] s
            WHERE (s.homeTeam_id IN ({home_id}, {away_id}) OR s.awayTeam_id IN ({home_id}, {away_id}))
                AND s.gameDate < '{as_of_date}'
                AND s.gameType = 2
                AND s.gameState IN ('OFF', 'FINAL')
            ORDER BY s.gameDate DESC
            """
            
            schedule = pd.read_sql(query_schedule, self.data_loader.conn)
            
            if len(schedule) == 0:
                logger.error("No historical games found")
                return None
            
            # Sort chronologically (oldest first)
            schedule = schedule.sort_values('gameDate').reset_index(drop=True)
            
            # ✅ Get the specific list of Game IDs
            game_ids = schedule['game_id'].unique().tolist()
            logger.info(f"  Identified {len(game_ids)} historical games for feature engineering")

            # ============================================================
            # 2. Use Data Loader for the heavy lifting
            # ============================================================
            # This reuses the logic from loaders.py, keeping things DRY
            
            team_game_xg = self.data_loader.load_team_game_xg(game_ids=game_ids)
            shot_xg = self.data_loader.load_shot_xg(game_ids=game_ids)
            events = self.data_loader.load_play_events(game_ids=game_ids)
            
            # ============================================================
            # 3. Normalize data types (Safety check)
            # ============================================================
            # Although loaders usually handle this, we enforce int explicitly 
            # to prevent the merge explosion you saw earlier.
            
            schedule['game_id'] = schedule['game_id'].astype(int)
            schedule['gameDate'] = pd.to_datetime(schedule['gameDate'])
            
            # Fix Team IDs in schedule
            schedule['homeTeam_id'] = schedule['homeTeam_id'].astype(int)
            schedule['awayTeam_id'] = schedule['awayTeam_id'].astype(int)

            # Fix IDs in fetched data
            if not team_game_xg.empty:
                team_game_xg['game_id'] = team_game_xg['game_id'].astype(int)
                team_game_xg['team_id'] = team_game_xg['team_id'].astype(int)
            
            if not shot_xg.empty:
                shot_xg['game_id'] = shot_xg['game_id'].astype(int)
                shot_xg['event_owner_team_id'] = shot_xg['event_owner_team_id'].fillna(-1).astype(int)
                
            if not events.empty:
                events['game_id'] = events['game_id'].astype(int)
                events['event_owner_team_id'] = events['event_owner_team_id'].fillna(-1).astype(int)
            
            logger.info(f"    Schedule: {len(schedule)} games")
            logger.info(f"    Team XG: {len(team_game_xg)} records")
            logger.info(f"    Shots: {len(shot_xg)} shots")
            logger.info(f"    Events: {len(events)} events")
            
            return {
                'schedule': schedule,
                'shot_xg': shot_xg,
                'team_game_xg': team_game_xg,
                'events': events
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_overtime(self, game_row: pd.Series) -> bool:
        """Detect if a game went to overtime/shootout"""
        if 'periodDescriptor_json' not in game_row:
            return False
        
        period_descriptor = game_row['periodDescriptor_json']
        
        if pd.isna(period_descriptor):
            return False
        
        try:
            if isinstance(period_descriptor, dict):
                period_type = period_descriptor.get('periodType', '')
                return period_type in ['OT', 'SO']
            
            elif isinstance(period_descriptor, str):
                try:
                    import json
                    period_data = json.loads(period_descriptor)
                    period_type = period_data.get('periodType', '')
                    return period_type in ['OT', 'SO']
                except json.JSONDecodeError:
                    return 'OT' in period_descriptor or 'SO' in period_descriptor
            
            return False
        except Exception as e:
            logger.warning(f"Error detecting OT: {e}")
            return False
    
    def predict_todays_games(self, bankroll: float = 1000) -> pd.DataFrame:
        """
        Predict all of today's games with quantitative betting decisions
        """
        # Initialize quantitative betting engine with calibration
        betting_engine = QuantitativeBettingEngine(
            bankroll=bankroll,
            max_portfolio_leverage=0.25,  # Max 25% of bankroll at risk
            min_edge_threshold=0.01,       # Minimum 1% edge required
            calibration_file='src/models/saved/threshold_analysis.csv'
        )
        
        # Fetch today's games
        games = self.fetch_todays_games()
        
        if len(games) == 0:
            logger.info("No games today or unable to fetch games")
            return pd.DataFrame()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PREDICTING {len(games)} GAMES WITH QUANTITATIVE BETTING ENGINE")
        logger.info(f"{'='*80}\n")
        
        predictions = []
        validation_failures = []
        all_features_rows = []
        
        for idx, game in games.iterrows():
            try:
                logger.info(f"\n[{idx+1}/{len(games)}] Processing: {game['awayTeam_abbrev']} @ {game['homeTeam_abbrev']}")
                
                # Predict with validation
                result = self.predict_with_validation(game)
                
                if not result['valid']:
                    validation_failures.append({
                        'game': result['game_info'],
                        'error': result['error']
                    })
                    logger.warning(f"  ⚠️  Skipping - {result['error']}")
                    continue
                
                # Capture features for debugging
                debug_row = result['features'].copy()
                debug_row.insert(0, 'matchup', f"{game['awayTeam_abbrev']} @ {game['homeTeam_abbrev']}")
                debug_row.insert(1, 'home_team', game['homeTeam_abbrev'])
                debug_row.insert(2, 'away_team', game['awayTeam_abbrev'])
                all_features_rows.append(debug_row)

                # Extract prediction
                home_prob = result['home_prob']
                away_prob = result['away_prob']
                
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
                
                # Get odds
                decimal_odds, american_odds = self._get_team_odds(game, team_type)
                
                # Make betting decision with new engine
                betting_decision = betting_engine.make_betting_decision(
                    game_id=str(game['game_id']),
                    team=predicted_team,
                    bet_type='moneyline',
                    model_probability=win_prob,
                    decimal_odds=decimal_odds
                )
                
                # Log decision
                if betting_decision['decision'] == 'BET':
                    logger.info(f"  ✅ BET: {predicted_team} ({win_prob:.1%})")
                    logger.info(f"     Edge: {betting_decision['edge']:+.1%}")
                    logger.info(f"     Calibrated Accuracy: {betting_decision['empirical_accuracy']:.1%}")
                    logger.info(f"     Bet: ${betting_decision['bet_size']:.2f} ({betting_decision['bet_size_pct']:.2f}%)")
                else:
                    logger.info(f"  ❌ NO BET: {predicted_team} ({win_prob:.1%})")
                    logger.info(f"     Reason: {betting_decision['reason']}")
                    logger.info(f"     Edge: {betting_decision['edge']:+.1%}")
                
                # Collect all edges for dynamic classification
                all_edges = [betting_engine.make_betting_decision(
                    game_id=str(g['game_id']),
                    team='temp',
                    bet_type='moneyline', 
                    model_probability=0.5,
                    decimal_odds=2.0
                )['edge'] for _, g in games.iterrows()]
                
                # Build prediction record with proper field mapping
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
                    
                    # Calibration info
                    'model_probability': win_prob,
                    'empirical_accuracy': betting_decision.get('empirical_accuracy', win_prob),
                    'accuracy_std_error': betting_decision.get('accuracy_std_error', 0),
                    
                    # Betting decision
                    'action': 'BET' if betting_decision['decision'] == 'BET' else 'NO BET',
                    'bet_size': betting_decision.get('bet_size', 0),
                    'bet_pct_bankroll': betting_decision.get('bet_size_pct', 0),
                    'kelly_fraction': betting_decision.get('kelly_fraction', 0),
                    
                    # Edge and value (edge_class will be set after all predictions)
                    'edge': betting_decision['edge'],
                    'edge_class': '',  # Will be set dynamically after collecting all edges
                    'expected_value': betting_decision.get('expected_value', 0),
                    'expected_roi': (betting_decision.get('expected_value', 0) / betting_decision.get('bet_size', 1) * 100) 
                                    if betting_decision.get('bet_size', 0) > 0 else 0,
                    'sharpe_ratio': betting_decision.get('sharpe_ratio', 0),
                    
                    # Odds
                    'decimal_odds': decimal_odds,
                    'american_odds': american_odds,
                    'implied_probability': betting_decision.get('implied_probability', 1/decimal_odds),
                    
                    # Reasoning
                    'reasoning': betting_decision.get('reason', 
                                f"Edge: {betting_decision['edge']:+.1%}, Calibrated: {betting_decision.get('empirical_accuracy', win_prob):.1%}"),
                    'validation_passed': True,
                    
                    # Portfolio info
                    'portfolio_correlation': betting_decision.get('portfolio_correlation', 0),
                    'current_portfolio_exposure': betting_decision.get('current_portfolio_exposure', 0)
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"  ❌ Error predicting game: {e}")
                import traceback
                traceback.print_exc()
                
                validation_failures.append({
                    'game': f"{game['awayTeam_abbrev']} @ {game['homeTeam_abbrev']}",
                    'error': str(e)
                })
        
        predictions_df = pd.DataFrame(predictions)
        
        # Dynamic edge classification based on actual distribution
        if len(predictions_df) > 0 and 'edge' in predictions_df.columns:
            # Calculate edge classifications based on the distribution
            all_edges = predictions_df['edge'].tolist()
            predictions_df['edge_class'] = predictions_df['edge'].apply(
                lambda e: self._classify_edge_dynamic(e, all_edges)
            )
        
        # Save debug features
        if all_features_rows:
            try:
                debug_features_df = pd.concat(all_features_rows, ignore_index=True)
                debug_filename = f"debug_features_{datetime.now().strftime('%Y%m%d')}.csv"
                debug_features_df.to_csv(debug_filename, index=False)
                logger.info(f"\n💾 DEBUG: Full feature set saved to {debug_filename}")
            except Exception as e:
                logger.error(f"Failed to save debug features: {e}")

        # Get portfolio metrics
        portfolio = betting_engine.get_portfolio_metrics()
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"PREDICTION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"  Total Games: {len(games)}")
        logger.info(f"  Successful Predictions: {len(predictions_df)}")
        logger.info(f"  Validation Failures: {len(validation_failures)}")
        
        if portfolio['position_count'] > 0:
            logger.info(f"\nPORTFOLIO METRICS:")
            logger.info(f"  Total Exposure: ${portfolio['total_exposure']:.2f} ({portfolio['exposure_pct']:.1f}%)")
            logger.info(f"  Average Edge: {portfolio['avg_edge']:.2%}")
            logger.info(f"  Portfolio EV: ${portfolio['portfolio_expected_value']:.2f}")
            logger.info(f"  Portfolio Sharpe: {portfolio['portfolio_sharpe']:.3f}")
        
        if validation_failures:
            logger.warning(f"\n⚠️  VALIDATION FAILURES:")
            for failure in validation_failures:
                logger.warning(f"  - {failure['game']}: {failure['error']}")
        
        logger.info(f"{'='*80}\n")
        
        return predictions_df
    
    def print_betting_card(self, predictions_df: pd.DataFrame):
        """Print quantitative betting card with calibration info"""
        print("\n" + "="*100)
        print(f"🏒 NHL QUANTITATIVE BETTING CARD - {self._get_today_mt()}")
        print("="*100)
        
        if len(predictions_df) == 0:
            print("\n❌ No games today or no predictions available")
            return
        
        # Sort by bet size (highest first), then by edge
        predictions_df = predictions_df.sort_values(
            ['bet_size', 'edge'], 
            ascending=[False, False]
        )
        
        # Group by action
        bet_games = predictions_df[predictions_df['action'] == 'BET']
        no_bet_games = predictions_df[predictions_df['action'] == 'NO BET']
        
        print(f"\n📊 SUMMARY")
        print(f"  Total Games: {len(predictions_df)}")
        print(f"  Games with Positive Edge: {len(bet_games)}")
        print(f"  Games to Skip: {len(no_bet_games)}")
        
        # Print bets (sorted by size)
        if len(bet_games) > 0:
            print("\n" + "-"*100)
            print("✅ BETTING RECOMMENDATIONS (Quantitative Analysis)")
            print("-"*100)
            
            for idx, pred in bet_games.iterrows():
                mt_time = self._convert_to_mountain_time(pred['game_time'])
                
                # Determine bet strength based on size
                if pred['bet_size'] >= pred['bet_pct_bankroll'] * 40:  # Top tier bets
                    print(f"\n🔥 STRONG BET: {pred['matchup']} - {mt_time}")
                else:
                    print(f"\n📊 BET: {pred['matchup']} - {mt_time}")
                
                print(f"   Venue: {pred['venue']}")
                print(f"   Pick: {pred['predicted_team']} ({pred['predicted_winner']})")
                
                # Show calibration adjustment
                calibration_diff = pred['empirical_accuracy'] - pred['model_probability']
                print(f"   Model Probability: {pred['model_probability']:.1%}")
                print(f"   Calibrated Accuracy: {pred['empirical_accuracy']:.1%} ({calibration_diff:+.1%} adjustment)")
                
                print(f"   Implied Probability: {pred['implied_probability']:.1%}")
                print(f"   💎 Edge: {pred['edge']:+.1%} ({pred['edge_class']})")
                print(f"   Odds: {pred['american_odds']:+d} (Decimal: {pred['decimal_odds']:.2f})")
                
                # Betting recommendation
                print(f"   💰 RECOMMENDED BET: ${pred['bet_size']:.2f} ({pred['bet_pct_bankroll']:.2f}% of bankroll)")
                print(f"   Kelly Fraction: {pred['kelly_fraction']:.4f}")
                print(f"   📈 Expected Value: ${pred['expected_value']:+.2f} | ROI: {pred['expected_roi']:+.1f}%")
                print(f"   Sharpe Ratio: {pred['sharpe_ratio']:.3f}")
                print(f"   {'-'*90}")
        
        # Print no-bet games
        if len(no_bet_games) > 0:
            print("\n" + "-"*100)
            print("⚠️  GAMES TO SKIP (Insufficient Edge)")
            print("-"*100)
            
            for idx, pred in no_bet_games.iterrows():
                mt_time = self._convert_to_mountain_time(pred['game_time'])
                print(f"\n   {pred['matchup']} - {mt_time}")
                print(f"   Lean: {pred['predicted_team']} ({pred['model_probability']:.1%})")
                print(f"   Calibrated: {pred['empirical_accuracy']:.1%} | Edge: {pred['edge']:+.1%}")
                print(f"   ❌ Reason: {pred['reasoning']}")
        
        print("\n" + "="*100)
        
        # Portfolio summary
        if len(bet_games) > 0:
            total_bet = bet_games['bet_size'].sum()
            total_ev = bet_games['expected_value'].sum()
            avg_model_prob = bet_games['model_probability'].mean()
            avg_calibrated = bet_games['empirical_accuracy'].mean()
            avg_edge = bet_games['edge'].mean()
            avg_sharpe = bet_games['sharpe_ratio'].mean()
            
            print("\n💵 PORTFOLIO SUMMARY:")
            print(f"  Total Stake: ${total_bet:.2f}")
            print(f"  Number of Bets: {len(bet_games)}")
            print(f"\n  📊 PROBABILITIES:")
            print(f"    Average Model Probability: {avg_model_prob:.1%}")
            print(f"    Average Calibrated Accuracy: {avg_calibrated:.1%}")
            print(f"    Calibration Adjustment: {(avg_calibrated - avg_model_prob)*100:+.1f}%")
            print(f"\n  💎 EDGE & VALUE:")
            print(f"    Average Edge: {avg_edge:+.2%}")
            print(f"    Total Expected Value: ${total_ev:+.2f}")
            print(f"    EV per Dollar: ${(total_ev/total_bet):+.3f}" if total_bet > 0 else "    N/A")
            print(f"    Portfolio ROI: {(total_ev/total_bet*100):+.1f}%" if total_bet > 0 else "    N/A")
            print(f"    Average Sharpe Ratio: {avg_sharpe:.3f}")
            
            # Edge distribution
            print(f"\n  📈 EDGE DISTRIBUTION:")
            edge_dist = bet_games.groupby('edge_class')['bet_size'].agg(['count', 'sum']).round(2)
            for edge_class in ['MASSIVE', 'LARGE', 'GOOD', 'DECENT', 'SMALL', 'MINIMAL']:
                if edge_class in edge_dist.index:
                    count = int(edge_dist.loc[edge_class, 'count'])
                    stake = edge_dist.loc[edge_class, 'sum']
                    print(f"    {edge_class}: {count} bet(s), ${stake:.2f} staked")
            
            print("\n" + "="*100)
        else:
            print("\n⚠️  No positive-edge opportunities found after calibration!")
            print("="*100)
    
    def run(self, bankroll: float = 1000):
        """Main execution method"""
        logger.info("Starting NHL Today's Games Predictor")
        
        # Load model
        self.load_model()
        
        # Initialize data loader
        self.initialize_data_loader()
        
        # Predict today's games (now includes validation)
        predictions = self.predict_todays_games(bankroll=bankroll)  # ✅ CORRECT
        
        # Print betting card
        self.print_betting_card(predictions)
        
        # Save predictions
        if len(predictions) > 0:
            output_file = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            predictions.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
        
        # Cleanup
        self.data_loader.disconnect()
        
        return predictions

def consolidate_duplicate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans for _x and _y columns, checks for zeros, and merges them 
    into a single standardized column.
    """
    # Identify all columns ending in _x
    x_cols = [col for col in df.columns if col.endswith('_x')]
    
    logger.info(f"Found {len(x_cols)} duplicate feature pairs to consolidate...")
    
    for col_x in x_cols:
        # Construct the corresponding _y name and base name
        base_name = col_x[:-2]  # remove '_x'
        col_y = base_name + '_y'
        
        if col_y in df.columns:
            # Check if columns are effectively zero
            # (We use a small threshold or exact 0 check)
            is_x_empty = (df[col_x] == 0).all()
            is_y_empty = (df[col_y] == 0).all()
            
            if is_x_empty and not is_y_empty:
                # Case 1: X is empty, Y has data -> Keep Y
                df[base_name] = df[col_y]
                # logger.info(f"  {base_name}: Kept _y (source 2), dropped _x (empty)")
                
            elif is_y_empty and not is_x_empty:
                # Case 2: Y is empty, X has data -> Keep X
                df[base_name] = df[col_x]
                # logger.info(f"  {base_name}: Kept _x (source 1), dropped _y (empty)")
                
            elif is_x_empty and is_y_empty:
                # Case 3: Both are empty -> Keep 0s
                df[base_name] = 0
                # logger.info(f"  {base_name}: Both empty, keeping 0s")
                
            else:
                # Case 4: Both have data (Conflict) -> Take the Max (or Mean)
                # Usually max is safer if one source might be missing data points
                df[base_name] = df[[col_x, col_y]].max(axis=1)
                # logger.info(f"  {base_name}: Merged _x and _y (max value)")
            
            # Drop the original _x and _y columns
            df.drop(columns=[col_x, col_y], inplace=True)
            
    return df

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
        model_dir='src/models/saved',
        connection_string=connection_string
    )
    
    # Run predictions with $1000 bankroll
    predictions = predictor.run(bankroll=1000)


if __name__ == "__main__":
    main()