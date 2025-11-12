import pandas as pd
import pyodbc
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class NHLDataLoader:
    """Load and manage NHL data from SQL Server database"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = pyodbc.connect(self.connection_string)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise
            
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def load_schedule(self, 
                     start_season: int = 20092010,
                     end_season: int = 20242025,
                     game_types: List[int] = [2]) -> pd.DataFrame:
        """
        Load schedule data
        game_types: [2] = Regular season, [3] = Playoffs
        """
        query = f"""
        SELECT 
            game_id, season, gameType, gameDate, venue, neutralSite,
            startTimeUTC, gameState, gameScheduleState,
            awayTeam_id, awayTeam_abbrev, awayTeam_score,
            homeTeam_id, homeTeam_abbrev, homeTeam_score,
            tvBroadcasts_json, periodDescriptor_json, gameOutcome_json
        FROM [nhlDB].[schedule].[schedule]
        WHERE season BETWEEN {start_season} AND {end_season}
        AND gameType IN ({','.join(map(str, game_types))})
        AND gameState IN ('OFF', 'FINAL')
        ORDER BY gameDate, game_id
        """
        
        df = pd.read_sql(query, self.conn)
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        logger.info(f"Loaded {len(df)} games from schedule")
        return df
    
    def load_game_roster(self, game_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load roster data for specified games"""
        if game_ids:
            game_id_str = ','.join(map(str, game_ids))
            where_clause = f"WHERE game_id IN ({game_id_str})"
        else:
            where_clause = ""
            
        query = f"""
        SELECT 
            roster_record_id, game_id, season, player_id, team_id,
            first_name, last_name, sweater_number, position_code, headshot_url
        FROM [nhlDB].[playbyplay].[GAME_ROSTER]
        {where_clause}
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"Loaded roster data for {df['game_id'].nunique()} games")
        return df
    
    def load_play_events(self, game_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load play-by-play events"""
        if game_ids:
            game_id_str = ','.join(map(str, game_ids))
            where_clause = f"WHERE game_id IN ({game_id_str})"
        else:
            where_clause = ""
            
        query = f"""
        SELECT 
            event_record_id, game_id, season, event_id, sort_order,
            period_number, period_type, time_in_period, time_remaining,
            type_code, type_desc_key, situation_code, event_owner_team_id,
            x_coord, y_coord, zone_code, shot_type,
            shooting_player_id, goalie_in_net_id, away_sog, home_sog,
            scoring_player_id, scoring_player_total,
            assist1_player_id, assist1_player_total,
            assist2_player_id, assist2_player_total,
            away_score, home_score,
            faceoff_winning_player_id, faceoff_losing_player_id,
            hitting_player_id, hittee_player_id, blocking_player_id,
            penalty_type_code, penalty_desc_key, penalty_duration,
            penalty_committed_by_player_id, penalty_drawn_by_player_id,
            stoppage_reason, giveaway_takeaway_player_id, missed_shot_reason
        FROM [nhlDB].[playbyplay].[PLAY_EVENTS_COMPLETE]
        {where_clause}
        ORDER BY game_id, sort_order
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"Loaded {len(df)} play events for {df['game_id'].nunique()} games")
        return df
    
    def load_shot_xg(self, game_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load shot-level xG data"""
        if game_ids:
            game_id_str = ','.join(map(str, game_ids))
            where_clause = f"WHERE game_id IN ({game_id_str})"
        else:
            where_clause = ""
            
        query = f"""
        SELECT 
            game_id, season, event_id, sort_order, period_number, period_type,
            time_in_period, event_owner_team_id, shooting_player_id, goalie_in_net_id,
            shot_type, type_code, is_goal, x_coord, y_coord, distance, angle,
            situation_code, is_powerplay, is_shorthanded, is_even_strength,
            is_empty_net, score_diff, is_rebound, is_rush, is_slot, xG
        FROM [nhlDB].[playbyplay].[shot_xG]
        {where_clause}
        ORDER BY game_id, sort_order
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"Loaded {len(df)} shots with xG for {df['game_id'].nunique()} games")
        return df
    
    def load_team_game_xg(self, game_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load team-level game xG aggregates"""
        if game_ids:
            game_id_str = ','.join(map(str, game_ids))
            where_clause = f"WHERE game_id IN ({game_id_str})"
        else:
            where_clause = ""
            
        query = f"""
        SELECT 
            game_id, season, team_id, is_home, opponent_team_id,
            goalie_id, opponent_goalie_id,
            shots_for, shots_against, goals_for, goals_against,
            xG_for, xG_against, xG_diff,
            shooting_percentage, save_percentage,
            xG_for_5v5, xG_against_5v5, xG_for_PP, xG_against_PP
        FROM [nhlDB].[playbyplay].[team_game_xG]
        {where_clause}
        ORDER BY game_id, team_id
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"Loaded team xG data for {df['game_id'].nunique()} games")
        return df
    
    def load_all_for_modeling(self, 
                              start_season: int = 20092010,
                              end_season: int = 20242025) -> Dict[str, pd.DataFrame]:
        """Load all tables needed for modeling"""
        logger.info(f"Loading all data from season {start_season} to {end_season}")
        
        # Load schedule first to get game_ids
        schedule = self.load_schedule(start_season, end_season)
        game_ids = schedule['game_id'].tolist()
        
        # Load all related tables
        data = {
            'schedule': schedule,
            'roster': self.load_game_roster(game_ids),
            'events': self.load_play_events(game_ids),
            'shot_xg': self.load_shot_xg(game_ids),
            'team_game_xg': self.load_team_game_xg(game_ids)
        }
        
        logger.info("All data loaded successfully")
        return data
    
    def get_recent_games(self, 
                        team_id: int, 
                        as_of_date: datetime,
                        n_games: int = 10) -> pd.DataFrame:
        """Get last N games for a team before a specific date"""
        query = f"""
        SELECT TOP {n_games}
            game_id, season, gameDate, venue,
            CASE 
                WHEN homeTeam_id = {team_id} THEN 1 
                ELSE 0 
            END as is_home,
            CASE 
                WHEN homeTeam_id = {team_id} THEN awayTeam_id 
                ELSE homeTeam_id 
            END as opponent_id,
            CASE 
                WHEN homeTeam_id = {team_id} THEN homeTeam_score 
                ELSE awayTeam_score 
            END as team_score,
            CASE 
                WHEN homeTeam_id = {team_id} THEN awayTeam_score 
                ELSE homeTeam_score 
            END as opponent_score
        FROM [nhlDB].[schedule].[schedule]
        WHERE (homeTeam_id = {team_id} OR awayTeam_id = {team_id})
        AND gameDate < '{as_of_date.strftime('%Y-%m-%d')}'
        AND gameState IN ('OFF', 'FINAL')
        ORDER BY gameDate DESC
        """
        
        df = pd.read_sql(query, self.conn)
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        return df


class DataCache:
    """Cache processed features to disk for faster loading"""
    
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def save(self, data: pd.DataFrame, name: str):
        """Save dataframe to parquet"""
        filepath = f"{self.cache_dir}/{name}.parquet"
        data.to_parquet(filepath, compression='snappy')
        logger.info(f"Cached {name} to {filepath}")
    
    def load(self, name: str) -> Optional[pd.DataFrame]:
        """Load cached dataframe"""
        filepath = f"{self.cache_dir}/{name}.parquet"
        try:
            data = pd.read_parquet(filepath)
            logger.info(f"Loaded cached {name} from {filepath}")
            return data
        except FileNotFoundError:
            logger.warning(f"Cache file not found: {filepath}")
            return None
    
    def exists(self, name: str) -> bool:
        """Check if cache exists"""
        import os
        return os.path.exists(f"{self.cache_dir}/{name}.parquet")


class TargetEncoder:
    """Encode targets for multi-task learning with proper types"""
    
    def __init__(self):
        self.outcome_mapping = {
            'home_win': 0,
            'away_win': 1,
            'overtime': 2  # Could be OT or shootout
        }
    
    def encode_targets(self, schedule_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Encode game outcomes and scores
        
        Returns:
            Dictionary with properly typed arrays:
            - 'outcome': int64 (class indices)
            - 'home_score': float32 (goal counts)
            - 'away_score': float32 (goal counts)
        """
        outcomes = []
        
        for _, row in schedule_df.iterrows():
            home_score = row['homeTeam_score']
            away_score = row['awayTeam_score']
            
            # Determine outcome
            if home_score > away_score:
                outcomes.append(self.outcome_mapping['home_win'])
            elif away_score > home_score:
                outcomes.append(self.outcome_mapping['away_win'])
            else:
                # Tie shouldn't happen in modern NHL, but could be OT/SO
                outcomes.append(self.outcome_mapping['overtime'])
        
        return {
            'outcome': np.array(outcomes, dtype=np.int64),
            'home_score': schedule_df['homeTeam_score'].values.astype(np.float32),
            'away_score': schedule_df['awayTeam_score'].values.astype(np.float32)
        }
    
    def decode_outcome(self, outcome_idx: int) -> str:
        """Convert outcome index back to string"""
        reverse_mapping = {v: k for k, v in self.outcome_mapping.items()}
        return reverse_mapping[outcome_idx]