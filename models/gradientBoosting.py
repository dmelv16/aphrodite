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
        
        # Aggregate by game and team
        event_features = events_df.groupby(['game_id', 'event_owner_team_id']).apply(
            lambda x: pd.Series({
                # Possession proxies
                'faceoff_wins': (x['type_code'] == 'FACEOFF').sum(),
                'faceoffs_total': (x['type_code'] == 'FACEOFF').sum(),
                
                # Physical play
                'hits_delivered': (x['type_code'] == 'HIT').sum(),
                'blocked_shots': (x['type_code'] == 'BLOCKED_SHOT').sum(),
                
                # Turnovers
                'giveaways': (x['type_code'] == 'GIVEAWAY').sum(),
                'takeaways': (x['type_code'] == 'TAKEAWAY').sum(),
                
                # Penalties
                'penalties_taken': (x['type_code'] == 'PENALTY').sum(),
                'penalty_minutes': x[x['type_code'] == 'PENALTY']['penalty_duration'].sum(),
                
                # Zone control (approximate)
                'offensive_zone_events': (x['zone_code'] == 'O').sum(),
                'defensive_zone_events': (x['zone_code'] == 'D').sum(),
                'neutral_zone_events': (x['zone_code'] == 'N').sum(),
                
                # Shot attempts (Corsi proxy)
                'shot_attempts': (x['type_code'].isin(['SHOT', 'MISSED_SHOT', 'BLOCKED_SHOT'])).sum(),
                
                # Scoring chances by period
                'period_1_shots': ((x['period_number'] == 1) & (x['type_code'] == 'SHOT')).sum(),
                'period_2_shots': ((x['period_number'] == 2) & (x['type_code'] == 'SHOT')).sum(),
                'period_3_shots': ((x['period_number'] == 3) & (x['type_code'] == 'SHOT')).sum()
            })
        ).reset_index()
        
        event_features.rename(columns={'event_owner_team_id': 'team_id'}, inplace=True)
        
        # Calculate derived metrics
        event_features['faceoff_win_pct'] = (
            event_features['faceoff_wins'] / event_features['faceoffs_total'].replace(0, 1)
        )
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
        """Calculate goalie-specific performance metrics"""
        
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
        
        # Calculate saves above expected
        goalie_shots['saves'] = goalie_shots['shots_faced'] - goalie_shots['goals_allowed']
        goalie_shots['expected_goals_allowed'] = goalie_shots['xG_faced']
        goalie_shots['goals_saved_above_expected'] = (
            goalie_shots['expected_goals_allowed'] - goalie_shots['goals_allowed']
        )
        goalie_shots['save_percentage'] = (
            goalie_shots['saves'] / goalie_shots['shots_faced'].replace(0, 1)
        )
        
        return goalie_shots
    
    def calculate_matchup_features(self, 
                                   schedule_df: pd.DataFrame,
                                   lookback_games: int = 5) -> pd.DataFrame:
        """Calculate head-to-head matchup history"""
        
        matchups = []
        
        for idx, game in schedule_df.iterrows():
            home_id = game['homeTeam_id']
            away_id = game['awayTeam_id']
            game_date = game['gameDate']
            
            # Get previous meetings
            previous = schedule_df[
                (schedule_df['gameDate'] < game_date) &
                (
                    ((schedule_df['homeTeam_id'] == home_id) & (schedule_df['awayTeam_id'] == away_id)) |
                    ((schedule_df['homeTeam_id'] == away_id) & (schedule_df['awayTeam_id'] == home_id))
                )
            ].tail(lookback_games)
            
            if len(previous) > 0:
                # Calculate matchup stats
                home_wins = previous[
                    ((previous['homeTeam_id'] == home_id) & (previous['homeTeam_score'] > previous['awayTeam_score'])) |
                    ((previous['awayTeam_id'] == home_id) & (previous['awayTeam_score'] > previous['homeTeam_score']))
                ].shape[0]
                
                matchups.append({
                    'game_id': game['game_id'],
                    'matchup_games_played': len(previous),
                    'home_matchup_wins': home_wins,
                    'away_matchup_wins': len(previous) - home_wins,
                    'avg_total_goals_matchup': (
                        previous['homeTeam_score'].mean() + previous['awayTeam_score'].mean()
                    )
                })
            else:
                matchups.append({
                    'game_id': game['game_id'],
                    'matchup_games_played': 0,
                    'home_matchup_wins': 0,
                    'away_matchup_wins': 0,
                    'avg_total_goals_matchup': 5.5  # League average
                })
        
        return pd.DataFrame(matchups)
    
    def calculate_rolling_correlations(self, 
                                       df: pd.DataFrame,
                                       col1: str,
                                       col2: str,
                                       window: int = 10) -> pd.Series:
        """Calculate rolling correlation between two metrics"""
        return df.groupby('team_id').apply(
            lambda x: x[col1].rolling(window).corr(x[col2])
        ).reset_index(level=0, drop=True)
    
    def calculate_consistency_metrics(self, df: pd.DataFrame, windows: List[int] = [5, 10]) -> pd.DataFrame:
        """Calculate team consistency/volatility metrics"""
        result_df = df.copy()
        
        for window in windows:
            # Goal scoring consistency
            result_df[f'goal_scoring_cv_{window}'] = (
                result_df.groupby('team_id')['goals_for']
                .transform(lambda x: x.rolling(window).std() / x.rolling(window).mean())
            )
            
            # Performance consistency (points)
            result_df[f'performance_consistency_{window}'] = (
                result_df.groupby('team_id')['points']
                .transform(lambda x: 1 / (1 + x.rolling(window).std()))
            )
        
        return result_df
    
    def generate_all_advanced_features(self,
                                       schedule_df: pd.DataFrame,
                                       shot_xg_df: pd.DataFrame,
                                       team_game_xg_df: pd.DataFrame,
                                       events_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all advanced features"""
        logger.info("Generating advanced features...")
        
        # Calculate all feature types
        xg_features = self.calculate_xg_features(shot_xg_df, team_game_xg_df)
        event_features = self.calculate_event_based_features(events_df)
        goalie_features = self.calculate_goalie_features(shot_xg_df, team_game_xg_df)
        matchup_features = self.calculate_matchup_features(schedule_df)
        
        # Merge all features
        advanced_df = schedule_df.copy()
        
        # Merge home team features
        home_xg = xg_features.copy()
        home_xg.columns = ['home_' + col if col not in ['game_id', 'team_id'] else col for col in home_xg.columns]
        advanced_df = advanced_df.merge(
            home_xg[home_xg['team_id'] == advanced_df['homeTeam_id']],
            left_on=['game_id', 'homeTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left',
            suffixes=('', '_home')
        )
        
        # Merge away team features
        away_xg = xg_features.copy()
        away_xg.columns = ['away_' + col if col not in ['game_id', 'team_id'] else col for col in away_xg.columns]
        advanced_df = advanced_df.merge(
            away_xg[away_xg['team_id'] == advanced_df['awayTeam_id']],
            left_on=['game_id', 'awayTeam_id'],
            right_on=['game_id', 'team_id'],
            how='left',
            suffixes=('', '_away')
        )
        
        # Add matchup features
        advanced_df = advanced_df.merge(matchup_features, on='game_id', how='left')
        
        logger.info(f"Generated {len(advanced_df.columns)} advanced features")
        return advanced_df