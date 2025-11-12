"""
Complete NHL Game Prediction Pipeline
Multi-level predictions with GPU-accelerated models
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data.loaders import NHLDataLoader, DataCache
from features.temporal import TemporalFeatureEngine
from features.advanced import AdvancedFeatureEngine
from models.gradientBoosting import XGBoostModel, XGBoostEnsemble
from models.neural import MultiTaskNHLModel, MultiTaskTrainer
from models.stacking import StackingEnsemble
from evaluation.metrics import ModelEvaluator
from utils.gpu_utils import check_gpu_availability

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nhl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NHLPredictionPipeline:
    """Complete pipeline for NHL game prediction"""
    
    def __init__(self, config_path: str = 'config/model_config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cache = DataCache()
        self.temporal_engine = TemporalFeatureEngine(
            windows=self.config['features']['rolling_windows']
        )
        self.advanced_engine = AdvancedFeatureEngine()
        
        self.models = {}
        self.feature_columns = None
        
        logger.info("Pipeline initialized")
        logger.info(f"GPU Available: {check_gpu_availability()}")
    
    def load_data(self, connection_string: str, use_cache: bool = True):
        """Load all NHL data"""
        logger.info("Loading data...")
        
        if use_cache and self.cache.exists('processed_features'):
            logger.info("Loading from cache")
            self.data = self.cache.load('processed_features')
            return
        
        # Initialize data loader
        loader = NHLDataLoader(connection_string)
        loader.connect()
        
        # Load all tables
        raw_data = loader.load_all_for_modeling(
            start_season=20182019,
            end_season=20242025
        )
        
        loader.disconnect()
        
        self.raw_data = raw_data
        logger.info(f"Loaded {len(raw_data['schedule'])} games")
    
    def engineer_features(self):
        """Generate all features"""
        logger.info("Engineering features...")
        
        schedule = self.raw_data['schedule']
        shot_xg = self.raw_data['shot_xg']
        team_game_xg = self.raw_data['team_game_xg']
        events = self.raw_data['events']
        
        # Prepare team-level data for each game
        team_stats = []
        for _, game in schedule.iterrows():
            # Home team
            home_stats = team_game_xg[
                (team_game_xg['game_id'] == game['game_id']) & 
                (team_game_xg['team_id'] == game['homeTeam_id'])
            ]
            if not home_stats.empty:
                home_stats = home_stats.iloc[0].to_dict()
                home_stats['is_home'] = 1
                home_stats['won'] = 1 if game['homeTeam_score'] > game['awayTeam_score'] else 0
                home_stats['points'] = 2 if home_stats['won'] else (1 if 'OT' in str(game.get('periodDescriptor_json', '')) else 0)
                home_stats['goal_differential'] = game['homeTeam_score'] - game['awayTeam_score']
                home_stats['gameDate'] = game['gameDate']
                team_stats.append(home_stats)
            
            # Away team
            away_stats = team_game_xg[
                (team_game_xg['game_id'] == game['game_id']) & 
                (team_game_xg['team_id'] == game['awayTeam_id'])
            ]
            if not away_stats.empty:
                away_stats = away_stats.iloc[0].to_dict()
                away_stats['is_home'] = 0
                away_stats['won'] = 1 if game['awayTeam_score'] > game['homeTeam_score'] else 0
                away_stats['points'] = 2 if away_stats['won'] else (1 if 'OT' in str(game.get('periodDescriptor_json', '')) else 0)
                away_stats['goal_differential'] = game['awayTeam_score'] - game['homeTeam_score']
                away_stats['gameDate'] = game['gameDate']
                team_stats.append(away_stats)
        
        team_stats_df = pd.DataFrame(team_stats)
        
        # Generate temporal features
        temporal_features = self.temporal_engine.generate_all_temporal_features(
            schedule, team_stats_df
        )
        
        # Generate advanced features
        advanced_features = self.advanced_engine.generate_all_advanced_features(
            schedule, shot_xg, team_game_xg, events
        )
        
        # Merge all features at game level
        game_features = schedule[['game_id', 'season', 'gameDate', 'homeTeam_id', 'awayTeam_id', 
                                  'homeTeam_score', 'awayTeam_score']].copy()
        
        # For each game, get home and away team features
        home_temporal = temporal_features[temporal_features['team_id'] == temporal_features['homeTeam_id']]
        away_temporal = temporal_features[temporal_features['team_id'] == temporal_features['awayTeam_id']]
        
        # Rename columns to indicate home/away
        home_cols = [c for c in home_temporal.columns if c not in ['game_id', 'team_id', 'season', 'gameDate']]
        away_cols = [c for c in away_temporal.columns if c not in ['game_id', 'team_id', 'season', 'gameDate']]
        
        home_temporal_renamed = home_temporal[['game_id'] + home_cols].copy()
        home_temporal_renamed.columns = ['game_id'] + [f'home_{c}' for c in home_cols]
        
        away_temporal_renamed = away_temporal[['game_id'] + away_cols].copy()
        away_temporal_renamed.columns = ['game_id'] + [f'away_{c}' for c in away_cols]
        
        # Merge features
        game_features = game_features.merge(home_temporal_renamed, on='game_id', how='left')
        game_features = game_features.merge(away_temporal_renamed, on='game_id', how='left')
        game_features = game_features.merge(advanced_features, on='game_id', how='left')
        
        # Create targets
        game_features['home_win'] = (game_features['homeTeam_score'] > game_features['awayTeam_score']).astype(int)
        game_features['away_win'] = (game_features['awayTeam_score'] > game_features['homeTeam_score']).astype(int)
        game_features['outcome'] = game_features.apply(
            lambda x: 0 if x['home_win'] else (1 if x['away_win'] else 2), axis=1
        )  # 0=home win, 1=away win, 2=OT/tie
        
        # Remove rows with missing features
        game_features = game_features.dropna()
        
        self.data = game_features
        
        # Cache processed features
        self.cache.save(game_features, 'processed_features')
        
        logger.info(f"Feature engineering complete. Shape: {game_features.shape}")
    
    def prepare_train_test_split(self, test_size: float = 0.2):
        """Split data into train and test sets"""
        logger.info("Preparing train/test split...")
        
        # Sort by date
        self.data = self.data.sort_values('gameDate')
        
        # Use temporal split (most recent games as test)
        split_idx = int(len(self.data) * (1 - test_size))
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        
        # Define feature columns (exclude metadata and targets)
        exclude_cols = ['game_id', 'season', 'gameDate', 'homeTeam_id', 'awayTeam_id',
                       'homeTeam_score', 'awayTeam_score', 'home_win', 'away_win', 'outcome']
        
        self.feature_columns = [c for c in self.data.columns if c not in exclude_cols]
        
        # Prepare training data
        self.X_train = train_data[self.feature_columns]
        self.y_train_outcome = train_data['outcome'].values
        self.y_train_home_score = train_data['homeTeam_score'].values
        self.y_train_away_score = train_data['awayTeam_score'].values
        
        # Prepare test data
        self.X_test = test_data[self.feature_columns]
        self.y_test_outcome = test_data['outcome'].values
        self.y_test_home_score = test_data['homeTeam_score'].values
        self.y_test_away_score = test_data['awayTeam_score'].values
        
        logger.info(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        logger.info(f"Number of features: {len(self.feature_columns)}")
    
    def train_level_1_outcome_classifier(self):
        """Train Level 1: Game Outcome Prediction"""
        logger.info("=" * 80)
        logger.info("LEVEL 1: GAME OUTCOME CLASSIFICATION")
        logger.info("=" * 80)
        
        # Split train into train/val
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train_outcome, 
            test_size=0.2, random_state=42, stratify=self.y_train_outcome
        )
        
        # XGBoost model
        logger.info("Training XGBoost classifier...")
        xgb_model = XGBoostModel(
            task='multiclass',
            params=self.config['models']['xgboost']['params']
        )
        xgb_model.params['num_class'] = 3
        xgb_model.train(X_tr, y_tr, X_val, y_val)
        
        self.models['xgboost_outcome'] = xgb_model
        
        # Evaluate
        test_preds = xgb_model.predict(self.X_test)
        test_preds_class = np.argmax(test_preds, axis=1)
        
        from sklearn.metrics import accuracy_score, log_loss
        accuracy = accuracy_score(self.y_test_outcome, test_preds_class)
        logloss = log_loss(self.y_test_outcome, test_preds)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Log Loss: {logloss:.4f}")
        
        return xgb_model
    
    def train_level_2_score_prediction(self):
        """Train Level 2: Score Regression"""
        logger.info("=" * 80)
        logger.info("LEVEL 2: SCORE PREDICTION")
        logger.info("=" * 80)
        
        # Split train into train/val
        X_tr, X_val, y_tr_home, y_val_home = train_test_split(
            self.X_train, self.y_train_home_score,
            test_size=0.2, random_state=42
        )
        _, _, y_tr_away, y_val_away = train_test_split(
            self.X_train, self.y_train_away_score,
            test_size=0.2, random_state=42
        )
        
        # Multi-task neural network
        logger.info("Training Multi-Task Neural Network...")
        
        input_dim = len(self.feature_columns)
        model = MultiTaskNHLModel(
            input_dim=input_dim,
            shared_layers=self.config['models']['multi_task']['shared_layers'],
            task_heads=self.config['models']['multi_task']['task_heads'],
            dropout=self.config['models']['multi_task']['dropout']
        )
        
        trainer = MultiTaskTrainer(
            model,
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Prepare targets
        y_train_dict = {
            'outcome': self.y_train_outcome,
            'home_score': self.y_train_home_score,
            'away_score': self.y_train_away_score
        }
        
        # Note: This is simplified - in practice, split properly
        trainer.fit(
            self.X_train.values,
            y_train_dict,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            patience=self.config['training']['early_stopping_patience']
        )
        
        self.models['multitask_nn'] = trainer
        
        # Evaluate
        predictions = trainer.predict(self.X_test.values)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae_home = mean_absolute_error(self.y_test_home_score, predictions['home_score'])
        mae_away = mean_absolute_error(self.y_test_away_score, predictions['away_score'])
        
        logger.info(f"Home Score MAE: {mae_home:.4f}")
        logger.info(f"Away Score MAE: {mae_away:.4f}")
        
        return trainer
    
    def train_ensemble(self):
        """Train stacking ensemble of all models"""
        logger.info("=" * 80)
        logger.info("TRAINING STACKING ENSEMBLE")
        logger.info("=" * 80)
        
        # Collect base models
        base_models = [self.models['xgboost_outcome']]
        
        # Train additional models for diversity
        logger.info("Training LightGBM for ensemble...")
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            **self.config['models']['lightgbm']['params']
        )
        lgb_model.fit(self.X_train, self.y_train_outcome)
        base_models.append(lgb_model)
        
        # Create stacking ensemble
        stacking = StackingEnsemble(
            base_models=base_models,
            meta_learner='lightgbm',
            cv_folds=5,
            task='multiclass'
        )
        
        stacking.fit(self.X_train, self.y_train_outcome)
        
        self.models['stacking_ensemble'] = stacking
        
        # Evaluate
        test_preds = stacking.predict_proba(self.X_test)
        test_preds_class = np.argmax(test_preds, axis=1)
        
        from sklearn.metrics import accuracy_score, log_loss
        accuracy = accuracy_score(self.y_test_outcome, test_preds_class)
        logloss = log_loss(self.y_test_outcome, test_preds)
        
        logger.info(f"Ensemble Test Accuracy: {accuracy:.4f}")
        logger.info(f"Ensemble Test Log Loss: {logloss:.4f}")
        
        return stacking
    
    def save_models(self, output_dir: str = 'models/saved'):
        """Save all trained models"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {output_dir}")
        
        if 'xgboost_outcome' in self.models:
            self.models['xgboost_outcome'].save_model(f"{output_dir}/xgboost_outcome")
        
        if 'multitask_nn' in self.models:
            self.models['multitask_nn'].save_checkpoint(f"{output_dir}/multitask_nn.pt")
        
        # Save feature columns
        import joblib
        joblib.dump(self.feature_columns, f"{output_dir}/feature_columns.pkl")
        
        logger.info("All models saved successfully")
    
    def run_full_pipeline(self, connection_string: str):
        """Run complete training pipeline"""
        logger.info("=" * 80)
        logger.info("STARTING NHL PREDICTION PIPELINE")
        logger.info("=" * 80)
        
        # Load data
        self.load_data(connection_string, use_cache=False)
        
        # Engineer features
        self.engineer_features()
        
        # Prepare train/test split
        self.prepare_train_test_split()
        
        # Train models
        self.train_level_1_outcome_classifier()
        self.train_level_2_score_prediction()
        self.train_ensemble()
        
        # Save models
        self.save_models()
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)


def main():
    """Main execution function"""
    
    # Database connection string (update with your credentials)
    connection_string = (
        "DRIVER={SQL Server};"
        "SERVER=your_server;"
        "DATABASE=nhlDB;"
        "Trusted_Connection=yes;"
    )
    
    # Initialize and run pipeline
    pipeline = NHLPredictionPipeline()
    pipeline.run_full_pipeline(connection_string)


if __name__ == "__main__":
    main()