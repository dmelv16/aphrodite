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
                # Rolling mean - FIXED: added shift(1) to prevent data leakage
                result_df[f'{col}_rolling_{window}'] = (
                    result_df.groupby(group_col)[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
                )
                
                # Rolling std - FIXED: added shift(1)
                result_df[f'{col}_rolling_std_{window}'] = (
                    result_df.groupby(group_col)[col]
                    .transform(lambda x: x.rolling(window, min_periods=2).std().shift(1))
                )
                
                # Rolling trend - FIXED: added shift(1)
                result_df[f'{col}_trend_{window}'] = (
                    result_df.groupby(group_col)[col]
                    .transform(lambda x: self._calculate_trend(x, window).shift(1))
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
            # FIXED: added shift(1) to prevent data leakage
            result_df[f'{col}_ema'] = (
                result_df.groupby(group_col)[col]
                .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean().shift(1))
            )
        
        return result_df
    
    def calculate_form_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team form indicators (streaks, momentum)"""
        result_df = df.copy()
        
        # Win streak - FIXED: shift applied in _calculate_streak
        result_df['win_streak'] = (
            result_df.groupby('team_id')['won']
            .transform(lambda x: self._calculate_streak(x))
        )
        
        # Points in last 5 games - FIXED: added shift(1)
        result_df['points_last_5'] = (
            result_df.groupby('team_id')['points']
            .transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        )
        
        # Hot hand indicator - FIXED: added shift(1)
        result_df['hot_hand'] = (
            result_df.groupby('team_id')['goals_for']
            .transform(lambda x: (x.rolling(3).mean().shift(1) - x.expanding().mean().shift(1)).fillna(0))
        )
        
        return result_df
    
    def _calculate_streak(self, series: pd.Series) -> pd.Series:
        """Calculate current streak (positive for wins, negative for losses)"""
        streaks = []
        current_streak = 0
        
        for val in series:
            # Store PREVIOUS streak before updating with current game
            streaks.append(current_streak)
            
            if pd.isna(val):
                continue
                
            if val == 1:  # Win
                current_streak = current_streak + 1 if current_streak >= 0 else 1
            else:  # Loss
                current_streak = current_streak - 1 if current_streak <= 0 else -1
        
        return pd.Series(streaks, index=series.index)
    
    def calculate_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate days of rest between games"""
        result_df = df.copy()
        
        # This is already correct - diff() naturally looks at previous row
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
        """Calculate games played in last N days (OPTIMIZED: vectorized approach)"""
        result_df = df.copy()
        
        # Store original index
        original_index = result_df.index
        
        # Set date as index for time-based rolling
        result_df = result_df.set_index('gameDate')
        
        # Create a dummy column to count
        result_df['game_count'] = 1
        
        # Group by team, roll over the date index, and count
        # shift(1) ensures we only count games BEFORE the current game
        col_name = f'games_in_{window}_days'
        result_df[col_name] = (
            result_df.groupby('team_id')['game_count']
            .rolling(f'{window}D')
            .count()
            .shift(1)
            .reset_index(0, drop=True)  # Drop the team_id level from the index
        )
        
        # Restore original index and fill NaNs
        result_df = result_df.reset_index().set_index(original_index).sort_index()
        result_df[col_name] = result_df[col_name].fillna(0)  # Fill NaNs (e.g., first games)
        del result_df['game_count']
        
        return result_df
    
    def calculate_home_away_splits(self, df: pd.DataFrame, windows: List[int] = [10, 20]) -> pd.DataFrame:
        """Calculate separate rolling stats for home and away games"""
        result_df = df.copy()
        
        for window in windows:
            # FIXED: added shift(1) to both home and away calculations
            # Home performance
            home_mask = result_df['is_home'] == 1
            result_df.loc[home_mask, f'home_win_pct_{window}'] = (
                result_df[home_mask]
                .groupby('team_id')['won']
                .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            )
            
            # Away performance
            away_mask = result_df['is_home'] == 0
            result_df.loc[away_mask, f'away_win_pct_{window}'] = (
                result_df[away_mask]
                .groupby('team_id')['won']
                .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            )
            
            # Fill missing values with forward fill, then default
            result_df[f'home_win_pct_{window}'] = (
                result_df.groupby('team_id')[f'home_win_pct_{window}']
                .fillna(method='ffill')
                .fillna(0.5)
            )
            result_df[f'away_win_pct_{window}'] = (
                result_df.groupby('team_id')[f'away_win_pct_{window}']
                .fillna(method='ffill')
                .fillna(0.5)
            )
        
        return result_df
    
    def calculate_season_progression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features related to season progression"""
        result_df = df.copy()
        
        # Games played in season (this is fine - cumcount is inherently "up to but not including")
        result_df['games_played_season'] = (
            result_df.groupby(['team_id', 'season'])
            .cumcount() + 1
        )
        
        # Season progress percentage (assuming 82 games)
        result_df['season_progress'] = result_df['games_played_season'] / 82.0
        
        # Month of season (this is fine - it's a property of the game date itself)
        result_df['month'] = result_df['gameDate'].dt.month
        
        # Day of week (this is fine - it's a property of the game date itself)
        result_df['day_of_week'] = result_df['gameDate'].dt.dayofweek
        
        # Season phase
        result_df['season_phase'] = pd.cut(
            result_df['season_progress'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['early', 'mid', 'late', 'playoff_push']
        )
        
        return result_df
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and performance trajectory"""
        result_df = df.copy()
        
        # FIXED: added shift(1) to all momentum calculations
        # Goal differential momentum
        result_df['gd_momentum'] = (
            result_df.groupby('team_id')['goal_differential']
            .transform(lambda x: (x.rolling(5).mean() - x.rolling(15).mean()).shift(1))
        )
        
        # Recent vs longer-term performance
        result_df['recent_vs_longterm'] = (
            result_df.groupby('team_id')['points']
            .transform(lambda x: (x.rolling(5).mean() - x.rolling(20).mean()).shift(1))
        )
        
        # Acceleration (improving or declining)
        result_df['performance_acceleration'] = (
            result_df.groupby('team_id')['points']
            .transform(lambda x: (x.rolling(3).mean() - x.rolling(3).mean().shift(3)).shift(1))
        )
        
        return result_df
    
    def calculate_rolling_correlations(self, 
                                       df: pd.DataFrame,
                                       col1: str,
                                       col2: str,
                                       window: int = 10) -> pd.Series:
        """Calculate rolling correlation between two metrics"""
        # FIXED: Added shift(1) to prevent data leakage
        return (
            df.groupby('team_id')
            .apply(lambda x: x[col1].rolling(window).corr(x[col2]).shift(1))
            .reset_index(level=0, drop=True)
        )
    
    def calculate_consistency_metrics(self, df: pd.DataFrame, windows: List[int] = [5, 10]) -> pd.DataFrame:
        """Calculate team consistency/volatility metrics"""
        result_df = df.copy()
        
        for window in windows:
            # FIXED: Added shift(1) to prevent data leakage
            # Goal scoring consistency (coefficient of variation)
            result_df[f'goal_scoring_cv_{window}'] = (
                result_df.groupby('team_id')['goals_for']
                .transform(lambda x: (x.rolling(window).std() / x.rolling(window).mean()).shift(1))
            )
            
            # Performance consistency (points)
            result_df[f'performance_consistency_{window}'] = (
                result_df.groupby('team_id')['points']
                .transform(lambda x: (1 / (1 + x.rolling(window).std())).shift(1))
            )
        
        return result_df
    
    def generate_all_temporal_features(self, 
                                       schedule_df: pd.DataFrame,
                                       team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all temporal features"""
        logger.info("Generating temporal features...")
        
        # Merge schedule with team stats
        df = schedule_df.merge(team_stats_df, on=['game_id', 'team_id'], how='left')
        
        # CRITICAL: Sort by team and date before any temporal calculations
        df = df.sort_values(['team_id', 'gameDate']).reset_index(drop=True)
        
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
        df = self.calculate_consistency_metrics(df)
        
        logger.info(f"Generated {len(df.columns)} total features")
        return df