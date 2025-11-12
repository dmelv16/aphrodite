import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TemporalFeatureEngine:
    """Generate rolling and temporal features for NHL games"""
    
    def __init__(self, windows: List[int] = [5, 10, 20]):
        self.windows = windows
        
    def calculate_rolling_stats(self, 
                                df: pd.DataFrame,
                                group_col: str,
                                value_cols: List[str],
                                windows: List[int]) -> pd.DataFrame:
        """Calculate rolling statistics for specified columns"""
        result_df = df.copy()
        
        for window in windows:
            for col in value_cols:
                # Rolling mean
                result_df[f'{col}_rolling_{window}'] = (
                    result_df.groupby(group_col)[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                
                # Rolling std
                result_df[f'{col}_rolling_std_{window}'] = (
                    result_df.groupby(group_col)[col]
                    .transform(lambda x: x.rolling(window, min_periods=2).std())
                )
                
                # Rolling trend (simple linear)
                result_df[f'{col}_trend_{window}'] = (
                    result_df.groupby(group_col)[col]
                    .transform(lambda x: self._calculate_trend(x, window))
                )
        
        return result_df
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate simple linear trend over window"""
        def trend(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return series.rolling(window, min_periods=2).apply(trend, raw=False)
    
    def calculate_exponential_moving_average(self,
                                            df: pd.DataFrame,
                                            group_col: str,
                                            value_cols: List[str],
                                            alpha: float = 0.3) -> pd.DataFrame:
        """Calculate exponentially weighted moving average (gives more weight to recent games)"""
        result_df = df.copy()
        
        for col in value_cols:
            result_df[f'{col}_ema'] = (
                result_df.groupby(group_col)[col]
                .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
            )
        
        return result_df
    
    def calculate_form_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team form indicators (streaks, momentum)"""
        result_df = df.copy()
        
        # Win streak (positive = wins, negative = losses)
        result_df['win_streak'] = (
            result_df.groupby('team_id')['won']
            .transform(lambda x: self._calculate_streak(x))
        )
        
        # Points in last 5 games (2 for win, 1 for OT loss, 0 for regulation loss)
        result_df['points_last_5'] = (
            result_df.groupby('team_id')['points']
            .transform(lambda x: x.rolling(5, min_periods=1).sum())
        )
        
        # Hot hand indicator (goals scored above average in last 3 games)
        result_df['hot_hand'] = (
            result_df.groupby('team_id')['goals_for']
            .transform(lambda x: (x.rolling(3).mean() - x.expanding().mean()).fillna(0))
        )
        
        return result_df
    
    def _calculate_streak(self, series: pd.Series) -> pd.Series:
        """Calculate current streak (positive for wins, negative for losses)"""
        streaks = []
        current_streak = 0
        
        for val in series:
            if pd.isna(val):
                streaks.append(current_streak)
                continue
                
            if val == 1:  # Win
                current_streak = current_streak + 1 if current_streak >= 0 else 1
            else:  # Loss
                current_streak = current_streak - 1 if current_streak <= 0 else -1
            
            streaks.append(current_streak)
        
        return pd.Series(streaks, index=series.index)
    
    def calculate_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate days of rest between games"""
        result_df = df.copy()
        
        result_df['rest_days'] = (
            result_df.groupby('team_id')['gameDate']
            .diff()
            .dt.days
            .fillna(3)  # Default for first game
        )
        
        # Back-to-back indicator
        result_df['is_back_to_back'] = (result_df['rest_days'] <= 1).astype(int)
        
        # Well-rested indicator (3+ days)
        result_df['is_well_rested'] = (result_df['rest_days'] >= 3).astype(int)
        
        return result_df
    
    def calculate_schedule_density(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Calculate games played in last N days"""
        result_df = df.copy()
        
        result_df[f'games_in_{window}_days'] = (
            result_df.groupby('team_id')['gameDate']
            .transform(lambda x: self._count_games_in_window(x, window))
        )
        
        return result_df
    
    def _count_games_in_window(self, dates: pd.Series, window: int) -> pd.Series:
        """Count games in rolling date window"""
        counts = []
        for i, date in enumerate(dates):
            cutoff = date - timedelta(days=window)
            count = ((dates[:i] >= cutoff) & (dates[:i] < date)).sum()
            counts.append(count)
        return pd.Series(counts, index=dates.index)
    
    def calculate_home_away_splits(self, df: pd.DataFrame, windows: List[int] = [10, 20]) -> pd.DataFrame:
        """Calculate separate rolling stats for home and away games"""
        result_df = df.copy()
        
        for window in windows:
            # Home performance
            result_df[f'home_win_pct_{window}'] = (
                result_df[result_df['is_home'] == 1]
                .groupby('team_id')['won']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Away performance
            result_df[f'away_win_pct_{window}'] = (
                result_df[result_df['is_home'] == 0]
                .groupby('team_id')['won']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Fill missing values
            result_df[f'home_win_pct_{window}'].fillna(method='ffill', inplace=True)
            result_df[f'away_win_pct_{window}'].fillna(0.5, inplace=True)
            result_df[f'away_win_pct_{window}'].fillna(method='ffill', inplace=True)
            result_df[f'home_win_pct_{window}'].fillna(0.5, inplace=True)
        
        return result_df
    
    def calculate_season_progression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features related to season progression"""
        result_df = df.copy()
        
        # Games played in season
        result_df['games_played_season'] = (
            result_df.groupby(['team_id', 'season'])
            .cumcount() + 1
        )
        
        # Season progress percentage (assuming 82 games)
        result_df['season_progress'] = result_df['games_played_season'] / 82.0
        
        # Month of season
        result_df['month'] = result_df['gameDate'].dt.month
        
        # Day of week (potential effect)
        result_df['day_of_week'] = result_df['gameDate'].dt.dayofweek
        
        # Season phase (early, mid, late, playoff_push)
        result_df['season_phase'] = pd.cut(
            result_df['season_progress'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['early', 'mid', 'late', 'playoff_push']
        )
        
        return result_df
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and performance trajectory"""
        result_df = df.copy()
        
        # Goal differential momentum
        result_df['gd_momentum'] = (
            result_df.groupby('team_id')['goal_differential']
            .transform(lambda x: x.rolling(5).mean() - x.rolling(15).mean())
        )
        
        # Recent vs longer-term performance
        result_df['recent_vs_longterm'] = (
            result_df.groupby('team_id')['points']
            .transform(lambda x: x.rolling(5).mean() - x.rolling(20).mean())
        )
        
        # Acceleration (improving or declining)
        result_df['performance_acceleration'] = (
            result_df.groupby('team_id')['points']
            .transform(lambda x: x.rolling(3).mean() - x.rolling(3).mean().shift(3))
        )
        
        return result_df
    
    def generate_all_temporal_features(self, 
                                       schedule_df: pd.DataFrame,
                                       team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all temporal features"""
        logger.info("Generating temporal features...")
        
        # Merge schedule with team stats
        df = schedule_df.merge(team_stats_df, on=['game_id', 'team_id'], how='left')
        
        # Sort by team and date
        df = df.sort_values(['team_id', 'gameDate'])
        
        # Calculate all feature types
        stat_columns = ['goals_for', 'goals_against', 'xG_for', 'xG_against', 
                       'shots_for', 'shots_against', 'shooting_percentage', 'save_percentage']
        
        df = self.calculate_rolling_stats(df, 'team_id', stat_columns, self.windows)
        df = self.calculate_exponential_moving_average(df, 'team_id', stat_columns)
        df = self.calculate_form_indicators(df)
        df = self.calculate_rest_days(df)
        df = self.calculate_schedule_density(df)
        df = self.calculate_home_away_splits(df)
        df = self.calculate_season_progression(df)
        df = self.calculate_momentum_features(df)
        
        logger.info(f"Generated {len(df.columns)} total features")
        return df