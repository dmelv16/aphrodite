import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngine:
    """Generate advanced analytics features from xG and event data"""
    
    def __init__(self):
        pass
    
    def calculate_xg_features(self, 
                             shot_xg_df: pd.DataFrame,
                             team_game_xg_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced xG-based features"""
        
        # Shot quality metrics by game
        shot_quality = shot_xg_df.groupby(['game_id', 'event_owner_team_id']).agg({
            'xG': ['mean', 'std', 'sum', 'max'],
            'is_slot': 'mean',
            'is_rebound': 'mean',
            'is_rush': 'mean',
            'is_high_danger': lambda x: (x > 0.15).mean(),  # Custom high-danger threshold
            'distance': ['mean', 'std'],
            'angle': ['mean', 'std']
        }).reset_index()
        
        shot_quality.columns = ['_'.join(col).strip('_') for col in shot_quality.columns]
        shot_quality.rename(columns={
            'event_owner_team_id': 'team_id',
            'xG_mean': 'avg_shot_quality',
            'xG_std': 'shot_quality_variance',
            'xG_sum': 'total_xG',
            'xG_max': 'best_chance_xG',
            'is_slot_mean': 'pct_slot_shots',
            'is_rebound_mean': 'pct_rebound_shots',
            'is_rush_mean': 'pct_rush_shots',
            'is_high_danger_<lambda>': 'pct_high_danger_shots'
        }, inplace=True)
        
        # Situational xG efficiency
        situational = shot_xg_df.groupby(['game_id', 'event_owner_team_id']).apply(
            lambda x: pd.Series({
                'xG_5v5': x[x['is_even_strength'] == 1]['xG'].sum(),
                'xG_PP': x[x['is_powerplay'] == 1]['xG'].sum(),
                'xG_SH': x[x['is_shorthanded'] == 1]['xG'].sum(),
                'shots_5v5': (x['is_even_strength'] == 1).sum(),
                'shots_PP': (x['is_powerplay'] == 1).sum(),
                'shots_SH': (x['is_shorthanded'] == 1).sum()
            })
        ).reset_index()
        
        situational.rename(columns={'event_owner_team_id': 'team_id'}, inplace=True)
        
        # Merge all xG features
        xg_features = shot_quality.merge(situational, on=['game_id', 'team_id'], how='left')
        xg_features = xg_features.merge(team_game_xg_df, on=['game_id', 'team_id'], how='left')
        
        # Calculate over/under performance
        xg_features['goals_above_expected'] = xg_features['goals_for'] - xg_features['xG_for']
        xg_features['goals_against_above_expected'] = xg_features['goals_against'] - xg_features['xG_against']
        
        # PDO proxy (shooting% + save%)
        xg_features['pdo'] = xg_features['shooting_percentage'] + xg_features['save_percentage']
        
        return xg_features
    
    def calculate_event_based_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from play-by-play events"""
        
        # FIXED: Calculate total faceoffs per game FIRST
        total_faceoffs_game = (
            events_df[events_df['type_code'] == '502']
            .groupby('game_id')['event_id']
            .count()
            .to_frame('faceoffs_total_game')
        )
        
        # Aggregate by game and team
        event_features = events_df.groupby(['game_id', 'event_owner_team_id']).apply(
            lambda x: pd.Series({
                # Possession proxies
                'faceoff_wins': (x['type_code'] == '502').sum(),
                
                # Physical play
                'hits_delivered': (x['type_code'] == '503').sum(),
                'blocked_shots': (x['type_code'] == '508').sum(),
                
                # Turnovers
                'giveaways': (x['type_code'] == '504').sum(),
                'takeaways': (x['type_code'] == '525').sum(),
                
                # Penalties
                'penalties_taken': (x['type_code'] == '509').sum(),
                'penalty_minutes': x[x['type_code'] == '509']['penalty_duration'].sum(),
                
                # Zone control (approximate)
                'offensive_zone_events': (x['zone_code'] == 'O').sum(),
                'defensive_zone_events': (x['zone_code'] == 'D').sum(),
                'neutral_zone_events': (x['zone_code'] == 'N').sum(),
                
                # FIXED: Shot attempts (Fenwick proxy - removed BLOCKED_SHOT)
                'shot_attempts': (x['type_code'].isin(['506', '507'])).sum(),
                
                # Scoring chances by period
                'period_1_shots': ((x['period_number'] == 1) & (x['type_code'] == '506')).sum(),
                'period_2_shots': ((x['period_number'] == 2) & (x['type_code'] == '506')).sum(),
                'period_3_shots': ((x['period_number'] == 3) & (x['type_code'] == '506')).sum()
            })
        ).reset_index()
        
        event_features.rename(columns={'event_owner_team_id': 'team_id'}, inplace=True)
        
        # FIXED: Merge the total faceoffs per game
        event_features = event_features.merge(total_faceoffs_game, on='game_id', how='left')
        
        # FIXED: Calculate faceoff_win_pct using total game faceoffs
        event_features['faceoff_win_pct'] = (
            event_features['faceoff_wins'] / event_features['faceoffs_total_game'].replace(0, 1)
        )
        
        # Calculate derived metrics
        event_features['giveaway_takeaway_diff'] = (
            event_features['takeaways'] - event_features['giveaways']
        )
        event_features['zone_control_index'] = (
            event_features['offensive_zone_events'] / 
            (event_features['offensive_zone_events'] + event_features['defensive_zone_events']).replace(0, 1)
        )
        
        return event_features
    
    def calculate_goalie_features(self, 
                                  shot_xg_df: pd.DataFrame,
                                  team_game_xg_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate goalie-specific performance metrics (GSAx)"""
        
        # Shots faced by goalie
        goalie_shots = shot_xg_df.groupby(['game_id', 'goalie_in_net_id']).agg({
            'xG': ['sum', 'mean', 'count'],
            'is_goal': 'sum',
            'is_high_danger': lambda x: (x > 0.15).sum()
        }).reset_index()
        
        goalie_shots.columns = ['_'.join(col).strip('_') for col in goalie_shots.columns]
        goalie_shots.rename(columns={
            'goalie_in_net_id': 'goalie_id',
            'xG_sum': 'xG_faced',
            'xG_mean': 'avg_xG_faced',
            'xG_count': 'shots_faced',
            'is_goal_sum': 'goals_allowed',
            'is_high_danger_<lambda>': 'high_danger_shots_faced'
        }, inplace=True)
        
        # Calculate saves above expected (GSAx)
        goalie_shots['saves'] = goalie_shots['shots_faced'] - goalie_shots['goals_allowed']
        goalie_shots['expected_goals_allowed'] = goalie_shots['xG_faced']
        goalie_shots['goals_saved_above_expected'] = (
            goalie_shots['expected_goals_allowed'] - goalie_shots['goals_allowed']
        )
        goalie_shots['save_percentage'] = (
            goalie_shots['saves'] / goalie_shots['shots_faced'].replace(0, 1)
        )
        
        return goalie_shots
    
    def generate_all_advanced_features(self,
                                       schedule_df: pd.DataFrame,
                                       shot_xg_df: pd.DataFrame,
                                       team_game_xg_df: pd.DataFrame,
                                       events_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all advanced features and merge them properly"""
        logger.info("Generating advanced features...")
        
        # Calculate all feature types
        xg_features = self.calculate_xg_features(shot_xg_df, team_game_xg_df)
        event_features = self.calculate_event_based_features(events_df)
        goalie_features = self.calculate_goalie_features(shot_xg_df, team_game_xg_df)
        
        # Start with schedule
        advanced_df = schedule_df.copy()
        
        # === MERGE HOME TEAM FEATURES ===
        
        # Home xG features
        home_xg = xg_features.copy()
        home_xg.columns = ['home_' + col if col not in ['game_id', 'team_id'] else col for col in home_xg.columns]
        advanced_df = advanced_df.merge(
            home_xg,
            left_on=['game_id', 'homeTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        # Home event features
        home_events = event_features.copy()
        home_events.columns = ['home_' + col if col not in ['game_id', 'team_id'] else col for col in home_events.columns]
        advanced_df = advanced_df.merge(
            home_events,
            left_on=['game_id', 'homeTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        # Home goalie features (requires matching on goalie_id)
        # First, get home goalie IDs from team_game_xg_df
        home_goalie_map = team_game_xg_df[['game_id', 'team_id', 'goalie_id']].copy()
        home_goalie_map = home_goalie_map.rename(columns={'goalie_id': 'home_goalie_id'})
        advanced_df = advanced_df.merge(
            home_goalie_map,
            left_on=['game_id', 'homeTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        # Now merge goalie features
        home_goalie = goalie_features.copy()
        home_goalie.columns = ['home_' + col if col not in ['game_id', 'goalie_id'] else col for col in home_goalie.columns]
        advanced_df = advanced_df.merge(
            home_goalie,
            left_on=['game_id', 'home_goalie_id'],
            right_on=['game_id', 'goalie_id'],
            how='left'
        )
        advanced_df.drop('goalie_id', axis=1, inplace=True, errors='ignore')
        
        # === MERGE AWAY TEAM FEATURES ===
        
        # Away xG features
        away_xg = xg_features.copy()
        away_xg.columns = ['away_' + col if col not in ['game_id', 'team_id'] else col for col in away_xg.columns]
        advanced_df = advanced_df.merge(
            away_xg,
            left_on=['game_id', 'awayTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        # Away event features
        away_events = event_features.copy()
        away_events.columns = ['away_' + col if col not in ['game_id', 'team_id'] else col for col in away_events.columns]
        advanced_df = advanced_df.merge(
            away_events,
            left_on=['game_id', 'awayTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        # Away goalie features
        away_goalie_map = team_game_xg_df[['game_id', 'team_id', 'goalie_id']].copy()
        away_goalie_map = away_goalie_map.rename(columns={'goalie_id': 'away_goalie_id'})
        advanced_df = advanced_df.merge(
            away_goalie_map,
            left_on=['game_id', 'awayTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        advanced_df.drop('team_id', axis=1, inplace=True, errors='ignore')
        
        away_goalie = goalie_features.copy()
        away_goalie.columns = ['away_' + col if col not in ['game_id', 'goalie_id'] else col for col in away_goalie.columns]
        advanced_df = advanced_df.merge(
            away_goalie,
            left_on=['game_id', 'away_goalie_id'],
            right_on=['game_id', 'goalie_id'],
            how='left'
        )
        advanced_df.drop('goalie_id', axis=1, inplace=True, errors='ignore')
        
        logger.info(f"Generated {len(advanced_df.columns)} total advanced features")
        return advanced_df