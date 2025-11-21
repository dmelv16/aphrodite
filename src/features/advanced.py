import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngine:
    """Generate advanced analytics features from HISTORICAL xG and event data"""
    
    def __init__(self, windows: List[int] = [5, 10, 20]):
        self.windows = windows
    
    def calculate_historical_xg_features(self, 
                                        shot_xg_df: pd.DataFrame,
                                        team_game_xg_df: pd.DataFrame,
                                        schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ROLLING advanced xG-based features from historical games"""
        
        # First, calculate per-game statistics
        shot_quality = shot_xg_df.groupby(['game_id', 'event_owner_team_id']).agg({
            'xG': ['mean', 'std', 'sum', 'max'],
            'is_slot': 'mean',
            'is_rebound': 'mean',
            'is_rush': 'mean',
            'distance': ['mean', 'std'],
            'angle': ['mean', 'std']
        }).reset_index()
        
        shot_quality.columns = ['_'.join(col).strip('_') for col in shot_quality.columns]
        shot_quality.rename(columns={
            'event_owner_team_id': 'team_id',
            'xG_mean': 'avg_shot_quality_game',
            'xG_std': 'shot_quality_variance_game',
            'xG_sum': 'total_xG_game',
            'xG_max': 'best_chance_xG_game',
            'is_slot_mean': 'pct_slot_shots_game',
            'is_rebound_mean': 'pct_rebound_shots_game',
            'is_rush_mean': 'pct_rush_shots_game'
        }, inplace=True)
        
        # Situational xG per game
        situational = shot_xg_df.groupby(['game_id', 'event_owner_team_id']).apply(
            lambda x: pd.Series({
                'xG_5v5_game': x[x['is_even_strength'] == 1]['xG'].sum(),
                'xG_PP_game': x[x['is_powerplay'] == 1]['xG'].sum(),
                'xG_SH_game': x[x['is_shorthanded'] == 1]['xG'].sum(),
                'shots_5v5_game': (x['is_even_strength'] == 1).sum(),
                'shots_PP_game': (x['is_powerplay'] == 1).sum(),
                'shots_SH_game': (x['is_shorthanded'] == 1).sum()
            })
        ).reset_index()
        situational.rename(columns={'event_owner_team_id': 'team_id'}, inplace=True)
        
        # Merge per-game statistics
        per_game_stats = shot_quality.merge(situational, on=['game_id', 'team_id'], how='left')
        per_game_stats = per_game_stats.merge(
            team_game_xg_df[['game_id', 'team_id', 'goals_for', 'goals_against', 
                            'xG_for', 'xG_against', 'shooting_percentage', 'save_percentage']],
            on=['game_id', 'team_id'], 
            how='left'
        )
        
        # Add game date for sorting
        per_game_stats = per_game_stats.merge(
            schedule_df[['game_id', 'gameDate']], 
            on='game_id', 
            how='left'
        )
        
        # CRITICAL: Sort by team and date
        per_game_stats = per_game_stats.sort_values(['team_id', 'gameDate']).reset_index(drop=True)
        
        # NOW calculate rolling features (with shift to prevent leakage)
        rolling_features = per_game_stats.copy()
        
        stat_cols = [col for col in per_game_stats.columns 
                    if col not in ['game_id', 'team_id', 'gameDate']]
        
        for window in self.windows:
            for col in stat_cols:
                # Rolling mean
                rolling_features[f'{col}_rolling_{window}'] = (
                    rolling_features.groupby('team_id')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
                )
        
        # Calculate derived metrics from rolling features
        for window in self.windows:
            rolling_features[f'goals_above_expected_rolling_{window}'] = (
                rolling_features[f'goals_for_rolling_{window}'] - 
                rolling_features[f'xG_for_rolling_{window}']
            )
            rolling_features[f'pdo_rolling_{window}'] = (
                rolling_features[f'shooting_percentage_rolling_{window}'] + 
                rolling_features[f'save_percentage_rolling_{window}']
            )
        
        # Keep only game_id, team_id, and rolling features
        keep_cols = ['game_id', 'team_id'] + [col for col in rolling_features.columns 
                                               if 'rolling' in col or col in ['goals_above_expected', 'pdo']]
        rolling_features = rolling_features[keep_cols]
        
        return rolling_features
    
    def calculate_historical_event_features(self, 
                                           events_df: pd.DataFrame,
                                           schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ROLLING event-based features from historical games"""
        
        # Calculate per-game event statistics
        total_faceoffs_game = (
            events_df[events_df['type_code'] == '502']
            .groupby('game_id')['event_id']
            .count()
            .to_frame('faceoffs_total_game')
        )
        
        per_game_events = events_df.groupby(['game_id', 'event_owner_team_id']).apply(
            lambda x: pd.Series({
                'faceoff_wins_game': (x['type_code'] == '502').sum(),
                'hits_delivered_game': (x['type_code'] == '503').sum(),
                'blocked_shots_game': (x['type_code'] == '508').sum(),
                'giveaways_game': (x['type_code'] == '504').sum(),
                'takeaways_game': (x['type_code'] == '525').sum(),
                'penalties_taken_game': (x['type_code'] == '509').sum(),
                'penalty_minutes_game': x[x['type_code'] == '509']['penalty_duration'].sum(),
                'offensive_zone_events_game': (x['zone_code'] == 'O').sum(),
                'defensive_zone_events_game': (x['zone_code'] == 'D').sum(),
                'shot_attempts_game': (x['type_code'].isin(['506', '507'])).sum(),
            })
        ).reset_index()
        per_game_events.rename(columns={'event_owner_team_id': 'team_id'}, inplace=True)
        
        # Merge total faceoffs
        per_game_events = per_game_events.merge(total_faceoffs_game, on='game_id', how='left')
        per_game_events['faceoff_win_pct_game'] = (
            per_game_events['faceoff_wins_game'] / 
            per_game_events['faceoffs_total_game'].replace(0, 1)
        )
        
        # Add game date
        per_game_events = per_game_events.merge(
            schedule_df[['game_id', 'gameDate']], 
            on='game_id', 
            how='left'
        )
        
        # CRITICAL: Sort by team and date
        per_game_events = per_game_events.sort_values(['team_id', 'gameDate']).reset_index(drop=True)
        
        # Calculate rolling features
        rolling_events = per_game_events.copy()
        
        event_cols = [col for col in per_game_events.columns 
                     if col.endswith('_game') and col != 'faceoffs_total_game']
        
        for window in self.windows:
            for col in event_cols:
                rolling_events[f'{col}_rolling_{window}'] = (
                    rolling_events.groupby('team_id')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
                )
        
        # Keep only rolling features
        keep_cols = ['game_id', 'team_id'] + [col for col in rolling_events.columns if 'rolling' in col]
        rolling_events = rolling_events[keep_cols]
        
        return rolling_events
    
    def generate_all_advanced_features(self,
                                      schedule_df: pd.DataFrame,
                                      shot_xg_df: pd.DataFrame,
                                      team_game_xg_df: pd.DataFrame,
                                      events_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all HISTORICAL advanced features and merge properly"""
        logger.info("Generating historical advanced features...")
        
        # Calculate rolling features
        xg_rolling = self.calculate_historical_xg_features(shot_xg_df, team_game_xg_df, schedule_df)
        event_rolling = self.calculate_historical_event_features(events_df, schedule_df)
        
        # Start with schedule
        advanced_df = schedule_df[['game_id', 'homeTeam_id', 'awayTeam_id']].copy()
        
        # Merge home team rolling features
        home_xg = xg_rolling.copy()
        home_xg.columns = ['home_' + col if col not in ['game_id', 'team_id'] else col 
                           for col in home_xg.columns]
        advanced_df = advanced_df.merge(
            home_xg,
            left_on=['game_id', 'homeTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        home_events = event_rolling.copy()
        home_events.columns = ['home_' + col if col not in ['game_id', 'team_id'] else col 
                               for col in home_events.columns]
        advanced_df = advanced_df.merge(
            home_events,
            left_on=['game_id', 'homeTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        # Merge away team rolling features
        away_xg = xg_rolling.copy()
        away_xg.columns = ['away_' + col if col not in ['game_id', 'team_id'] else col 
                           for col in away_xg.columns]
        advanced_df = advanced_df.merge(
            away_xg,
            left_on=['game_id', 'awayTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        away_events = event_rolling.copy()
        away_events.columns = ['away_' + col if col not in ['game_id', 'team_id'] else col 
                               for col in away_events.columns]
        advanced_df = advanced_df.merge(
            away_events,
            left_on=['game_id', 'awayTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        # Cleanup
        advanced_df = advanced_df.drop(['homeTeam_id', 'awayTeam_id'], axis=1, errors='ignore')
        
        logger.info(f"Generated {len(advanced_df.columns)} total advanced features")
        return advanced_df