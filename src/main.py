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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

# Import custom modules
from src.data.loaders import NHLDataLoader, DataCache
from src.features.temporal import TemporalFeatureEngine
from src.features.advanced import AdvancedFeatureEngine
from src.models.gradientBoosting import XGBoostModel, XGBoostEnsemble
from src.models.poisson_neural import PoissonMultiTaskNHLModel, PoissonMultiTaskTrainer
from src.models.stacking import StackingEnsemble
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.improved_score_analyzer import ImprovedScorePredictionAnalyzer 
from src.utils.gpu_utils import check_gpu_availability

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
    
    def __init__(self, config_path: str = 'src/config/model_config.yaml'):
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
            start_season=20092010,
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

        try:
            logger.info("Normalizing join key data types...")
            
            # Convert schedule keys to integer
            schedule['game_id'] = schedule['game_id'].astype(int)
            schedule['homeTeam_id'] = schedule['homeTeam_id'].astype(int)
            schedule['awayTeam_id'] = schedule['awayTeam_id'].astype(int)
            
            # Convert team_game_xg keys to integer
            team_game_xg['game_id'] = team_game_xg['game_id'].astype(int)
            team_game_xg['team_id'] = team_game_xg['team_id'].astype(int)
            shot_xg['game_id'] = shot_xg['game_id'].astype(int)
            events['game_id'] = events['game_id'].astype(int)

            logger.info("Data types normalized successfully.")
        except Exception as e:
            logger.error(f"Failed to normalize data types: {e}")
            logger.error("This is likely the cause of the KeyError. Please check data loading.")
            raise
        
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
                home_stats['points'] = 2 if home_stats['won'] else (1 if detect_overtime_game(game.get('periodDescriptor_json')) else 0)
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
                away_stats['points'] = 2 if away_stats['won'] else (1 if detect_overtime_game(game.get('periodDescriptor_json')) else 0)
                away_stats['goal_differential'] = game['awayTeam_score'] - game['homeTeam_score']
                away_stats['gameDate'] = game['gameDate']
                team_stats.append(away_stats)
        
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
        
        # Generate temporal features
        temporal_features = self.temporal_engine.generate_all_temporal_features(
            schedule, team_stats_df
        )

        # Generate advanced features (already returns game-level with home_/away_ prefixes)
        advanced_features = self.advanced_engine.generate_all_advanced_features(
            schedule, shot_xg, team_game_xg, events
        )

        # Merge all features at game level
        game_features = schedule[['game_id', 'season', 'gameDate', 'homeTeam_id', 'awayTeam_id', 
                                'homeTeam_score', 'awayTeam_score']].copy()

        # Simple merge - no need to filter or rename!
        game_features = game_features.merge(temporal_features, on='game_id', how='left')
        game_features = game_features.merge(advanced_features, on='game_id', how='left')

        logger.info(f"  Shape before cleanup: {game_features.shape}")

        game_features = consolidate_duplicate_features(game_features)
        
        logger.info(f"  Shape after cleanup: {game_features.shape}")
        
        # === CREATE PROPER OUTCOME LABELS ===
        # First, create basic win indicators
        game_features['home_win'] = (game_features['homeTeam_score'] > game_features['awayTeam_score']).astype(int)
        game_features['away_win'] = (game_features['awayTeam_score'] > game_features['homeTeam_score']).astype(int)

        # Binary outcome: 0 = Away Win, 1 = Home Win
        game_features['outcome'] = game_features['home_win']
        
        # Detect OT games from schedule data
        logger.info("Detecting overtime/shootout games...")
        schedule['went_to_ot'] = schedule['periodDescriptor_json'].apply(detect_overtime_game)
        
        # Merge the OT indicator with game_features
        game_features = game_features.merge(
            schedule[['game_id', 'went_to_ot']], 
            on='game_id', 
            how='left'
        )
        
        # Fill any missing went_to_ot values with False
        game_features['went_to_ot'] = game_features['went_to_ot'].fillna(False).astype(bool)
        
        # Log outcome distribution
        logger.info("\n" + "="*80)
        logger.info("BINARY OUTCOME ENCODING")
        logger.info("="*80)
        logger.info(f"Total games: {len(game_features)}")
        logger.info(f"\nOutcome distribution:")
        outcome_counts = game_features['outcome'].value_counts().sort_index()
        for outcome_val in [0, 1]:
            count = outcome_counts.get(outcome_val, 0)
            outcome_name = {0: "Away Win", 1: "Home Win"}[outcome_val]
            pct = count / len(game_features) * 100 if len(game_features) > 0 else 0
            logger.info(f"  {outcome_val} ({outcome_name}): {count} ({pct:.1f}%)")

        logger.info(f"\nOf these, {game_features['went_to_ot'].sum()} ({game_features['went_to_ot'].sum()/len(game_features)*100:.1f}%) went to OT/SO")
        logger.info("="*80 + "\n")
        
        # Handle NaN values intelligently instead of dropping all rows
        logger.info("Handling missing values...")
        
        # Fill NaN values with appropriate defaults
        # For rolling stats and temporal features, fill with 0 or neutral values
        for col in game_features.columns:
            if game_features[col].isnull().any():
                if col not in ['game_id', 'season', 'gameDate', 'homeTeam_id', 'awayTeam_id',
                            'homeTeam_score', 'awayTeam_score', 'home_win', 'away_win', 'outcome', 'went_to_ot']:
                    # Fill numeric columns with 0
                    if game_features[col].dtype in ['float64', 'int64']:
                        game_features[col].fillna(0, inplace=True)
        
        # Check final state
        remaining_nan = game_features.isnull().sum().sum()
        logger.info(f"Remaining NaN values after filling: {remaining_nan}")
        
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
                    'homeTeam_score', 'awayTeam_score', 'home_win', 'away_win', 'outcome', 'went_to_ot']
        
        # Get all column names
        all_cols = self.data.columns.tolist()
        
        # Filter for potential feature columns
        potential_features = [c for c in all_cols if c not in exclude_cols]
        
        # Remove non-numeric columns
        numeric_features = []
        for col in potential_features:
            dtype = self.data[col].dtype
            if dtype in ['int64', 'float64', 'bool', 'int8', 'int16', 'int32', 
                        'float16', 'float32']:
                numeric_features.append(col)
            elif dtype.name == 'category':
                # Convert category to numeric codes
                logger.info(f"Converting category column to numeric: {col}")
                self.data[col] = self.data[col].cat.codes
                numeric_features.append(col)
            elif dtype == 'object':
                # Try to convert object columns to numeric
                try:
                    self.data[col] = pd.to_numeric(self.data[col])
                    numeric_features.append(col)
                    logger.info(f"Converted object column to numeric: {col}")
                except:
                    logger.warning(f"Dropping non-numeric column: {col} (dtype: {dtype})")
                    self.data.drop(col, axis=1, inplace=True)
            else:
                logger.warning(f"Dropping column with unsupported dtype: {col} (dtype: {dtype})")
        
        # CRITICAL: Filter out leaking features
        logger.info("Checking for data leakage...")
        safe_features, leaking_features = identify_leaking_features(self.data, numeric_features)
        
        if leaking_features:
            logger.warning(f"Found {len(leaking_features)} potentially leaking features:")
            for feat in leaking_features[:20]:  # Show first 20
                logger.warning(f"  - {feat}")
            logger.warning("These features will be EXCLUDED from modeling")
        
        self.feature_columns = safe_features
        
        # Update train/test data after column filtering
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        
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
        
        # DIAGNOSTIC: Check outcome distribution
        logger.info(f"\n{'='*80}")
        logger.info("OUTCOME DISTRIBUTION CHECK")
        logger.info(f"{'='*80}")
        logger.info(f"Training set outcomes:\n{pd.Series(self.y_train_outcome).value_counts().sort_index()}")
        logger.info(f"\nTest set outcomes:\n{pd.Series(self.y_test_outcome).value_counts().sort_index()}")
        logger.info(f"{'='*80}\n")
        
        # Calculate expected accuracy for random guessing
        train_outcome_dist = pd.Series(self.y_train_outcome).value_counts(normalize=True)
        expected_random_acc = train_outcome_dist.max()
        logger.info(f"Baseline (always predict most common class): {expected_random_acc:.4f}")
        logger.info(f"Expected accuracy for good model: 0.52-0.60")
        
        logger.info(f"\nTrain size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        logger.info(f"Number of features: {len(self.feature_columns)}")
        logger.info(f"Feature dtypes: {self.X_train.dtypes.value_counts().to_dict()}")
        
    def train_level_1_outcome_classifier(self):
        """
        Enhanced Level 1 training with threshold analysis
        (Replace the existing train_level_1_outcome_classifier method)
        """
        logger.info("=" * 80)
        logger.info("LEVEL 1: BINARY OUTCOME CLASSIFICATION (Home Win vs Away Win)")
        logger.info("=" * 80)
        
        from sklearn.model_selection import train_test_split
        
        # Split train into train/val
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train_outcome, 
            test_size=0.2, random_state=42, stratify=self.y_train_outcome
        )
        
        # XGBoost model - BINARY classification
        logger.info("Training XGBoost classifier...")
        
        xgb_model = XGBoostModel(
            task='binary',
            params=self.config['models']['xgboost']['params']
        )
        # Remove num_class for binary classification
        if 'num_class' in xgb_model.params:
            del xgb_model.params['num_class']
        
        xgb_model.train(X_tr, y_tr, X_val, y_val)
        
        self.models['xgboost_outcome'] = xgb_model
        
        # Evaluate
        test_preds_proba = xgb_model.predict(self.X_test)
        
        # For binary classification, predictions might be 1D or 2D
        if len(test_preds_proba.shape) == 1:
            test_preds_class = (test_preds_proba > 0.5).astype(int)
            test_probs = test_preds_proba
        else:
            test_preds_class = np.argmax(test_preds_proba, axis=1)
            test_probs = test_preds_proba[:, 1]
        
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
        
        logger.info(f"Classes in training set: {np.unique(self.y_train_outcome)}")
        logger.info(f"Classes in test set: {np.unique(self.y_test_outcome)}")
        
        accuracy = accuracy_score(self.y_test_outcome, test_preds_class)
        
        # Calculate log_loss for binary
        try:
            if len(test_preds_proba.shape) == 1:
                logloss = log_loss(self.y_test_outcome, test_preds_proba)
            else:
                logloss = log_loss(self.y_test_outcome, test_preds_proba)
        except Exception as e:
            logger.warning(f"Could not calculate log_loss: {e}")
            logloss = None
        
        # Calculate AUC-ROC
        try:
            auc = roc_auc_score(self.y_test_outcome, test_probs)
            logger.info(f"Test AUC-ROC: {auc:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc = None
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        if logloss:
            logger.info(f"Test Log Loss: {logloss:.4f}")
        
        # Calculate improvement over baseline
        home_win_rate = (self.y_train_outcome == 1).mean()
        baseline_accuracy = max(home_win_rate, 1 - home_win_rate)
        logger.info(f"Baseline (always predict most common): {baseline_accuracy:.4f}")
        logger.info(f"Improvement over baseline: {(accuracy - baseline_accuracy):.4f} ({(accuracy - baseline_accuracy)/baseline_accuracy*100:.1f}%)")
        
        # Additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(self.y_test_outcome, test_preds_class, target_names=['Away Win', 'Home Win'], zero_division=0)}")
        
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test_outcome, test_preds_class)
        logger.info(f"\n{cm}")
        logger.info("              Predicted")
        logger.info("              Away  Home")
        logger.info(f"Actual Away   {cm[0,0]:4d}  {cm[0,1]:4d}")
        logger.info(f"       Home   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # ============================================================================
        # NEW: COMPREHENSIVE THRESHOLD ANALYSIS
        # ============================================================================
        threshold_df = analyze_prediction_thresholds(self.y_test_outcome, test_probs)
        print_threshold_analysis(threshold_df, logger)
        save_threshold_analysis(threshold_df, output_dir='src/models/saved')
        
        # Store threshold analysis for later use
        self.threshold_analysis = threshold_df
        
        # Feature importance
        feature_importance = xgb_model.get_feature_importance()
        logger.info("\nTop 20 Most Important Features:")
        for idx, row in feature_importance.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return xgb_model
    
    def train_level_2_score_prediction_poisson(self):
        """
        UPDATED: Train Level 2 with Poisson Score Prediction
        """
        logger.info("=" * 80)
        logger.info("LEVEL 2: POISSON SCORE PREDICTION")
        logger.info("=" * 80)
        
        # Split train into train/val
        X_tr, X_val, y_tr_outcome, y_val_outcome = train_test_split(
            self.X_train, self.y_train_outcome,
            test_size=0.2, random_state=42, stratify=self.y_train_outcome
        )
        
        _, _, y_tr_home, y_val_home = train_test_split(
            self.X_train, self.y_train_home_score,
            test_size=0.2, random_state=42
        )
        
        _, _, y_tr_away, y_val_away = train_test_split(
            self.X_train, self.y_train_away_score,
            test_size=0.2, random_state=42
        )
        
        # CHANGED: Use Poisson Multi-task Neural Network
        logger.info("Training Poisson Multi-Task Neural Network...")
        
        input_dim = len(self.feature_columns)
        
        # Create Poisson model
        poisson_model = PoissonMultiTaskNHLModel(
            input_dim=input_dim,
            shared_layers=self.config['models']['multi_task']['shared_layers'],
            task_heads=self.config['models']['multi_task']['task_heads'],
            dropout=self.config['models']['multi_task']['dropout']
        )
        
        # Create Poisson trainer
        trainer = PoissonMultiTaskTrainer(
            poisson_model,
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            task_weights={
                'outcome': 1.0,
                'home_score': 0.5,
                'away_score': 0.5
            }
        )
        
        # Prepare training data
        y_train_dict = {
            'outcome': y_tr_outcome,
            'home_score': y_tr_home,
            'away_score': y_tr_away
        }
        
        y_val_dict = {
            'outcome': y_val_outcome,
            'home_score': y_val_home,
            'away_score': y_val_away
        }
        
        # Train with Poisson loss
        history = trainer.fit(
            X_tr.values,
            y_train_dict,
            X_val.values,
            y_val_dict,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            patience=self.config['training']['early_stopping_patience']
        )
        
        # Store the Poisson model
        self.models['poisson_multitask'] = trainer
        
        # CHANGED: Evaluate with Poisson predictions
        logger.info("\nEvaluating Poisson predictions...")
        predictions = trainer.predict_poisson_params(self.X_test.values)
        
        # Extract lambda parameters (expected values)
        expected_home = predictions['home_lambda']
        expected_away = predictions['away_lambda']
        
        from sklearn.metrics import mean_absolute_error
        mae_home = mean_absolute_error(self.y_test_home_score, expected_home)
        mae_away = mean_absolute_error(self.y_test_away_score, expected_away)
        mae_total = mean_absolute_error(
            self.y_test_home_score + self.y_test_away_score,
            expected_home + expected_away
        )
        
        logger.info(f"Home Score MAE: {mae_home:.4f}")
        logger.info(f"Away Score MAE: {mae_away:.4f}")
        logger.info(f"Total Score MAE: {mae_total:.4f}")
        
        # NEW: Calculate exact Over/Under probabilities for common lines
        logger.info("\nExact O/U Probabilities (Poisson):")
        from scipy import stats
        
        for line in [5.5, 6.0, 6.5]:
            over_probs = []
            
            for h_lambda, a_lambda in zip(expected_home, expected_away):
                # Calculate exact probability using Poisson
                prob_over = 0
                for h in range(15):  # Consider scores up to 14
                    for a in range(15):
                        if h + a > line:
                            prob_over += stats.poisson.pmf(h, h_lambda) * stats.poisson.pmf(a, a_lambda)
                over_probs.append(prob_over)
            
            over_probs = np.array(over_probs)
            actual_over = (self.y_test_home_score + self.y_test_away_score > line).mean()
            
            logger.info(f"  Line {line}: Predicted P(Over)={over_probs.mean():.3f}, "
                       f"Actual={actual_over:.3f}, "
                       f"Calibration Error={over_probs.mean() - actual_over:+.3f}")
        
        return trainer
    
    def analyze_score_predictions(self):
        """
        NEW: Comprehensive analysis of score predictions with Poisson
        """
        if 'poisson_multitask' not in self.models:
            logger.warning("Poisson model not found. Train it first.")
            return
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE SCORE & O/U ANALYSIS")
        logger.info("=" * 80)
        
        # Get Poisson predictions
        trainer = self.models['poisson_multitask']
        predictions = trainer.predict_poisson_params(self.X_test.values)
        
        # Create analyzer with Poisson predictions
        analyzer = ImprovedScorePredictionAnalyzer(
            self.y_test_home_score,
            self.y_test_away_score,
            predictions['home_lambda'],
            predictions['away_lambda'],
            prediction_type='poisson'  # Enable exact probability calculations
        )
        
        # Generate comprehensive report
        metrics, calibration_df, edges = analyzer.generate_enhanced_report()
        
        # Create visualizations
        analyzer.plot_enhanced_analysis(save_path='models/saved/poisson_analysis.png')
        
        # Find betting edges
        edges_df = analyzer.find_betting_edges_improved(
            min_edge=0.05,
            min_confidence=0.55,
            kelly_fraction=0.25
        )
        
        if not edges_df.empty:
            logger.info(f"\nFound {len(edges_df)} O/U betting opportunities")
            logger.info(f"Expected ROI: {edges_df['roi'].mean():.3f}")
            
            # Save edges to CSV
            edges_df.to_csv('models/saved/ou_betting_edges.csv', index=False)
            logger.info("Betting edges saved to models/saved/ou_betting_edges.csv")
        
        # Store analyzer for later use
        self.score_analyzer = analyzer
        
        return analyzer
    
    def train_ensemble(self):
        """
        Enhanced ensemble training that stores LightGBM separately
        (Replace the existing train_ensemble method)
        """
        logger.info("=" * 80)
        logger.info("TRAINING STACKING ENSEMBLE (Binary)")
        logger.info("=" * 80)
        
        # Collect base models
        base_models = [self.models['xgboost_outcome']]
        
        # Train additional models for diversity
        logger.info("Training LightGBM for ensemble...")
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            **self.config['models']['lightgbm']['params']
        )
        lgb_model.fit(self.X_train, self.y_train_outcome)
        base_models.append(lgb_model)
        
        # NEW: Store LightGBM separately for saving
        self.models['lightgbm_outcome'] = lgb_model
        logger.info("  ✓ LightGBM model stored")
        
        stacking = StackingEnsemble(
            base_models=base_models,
            meta_learner='lightgbm',
            cv_folds=5,
            task='binary'
        )
        
        stacking.fit(self.X_train, self.y_train_outcome)
        
        self.models['stacking_ensemble'] = stacking
        
        # Evaluate
        test_preds = stacking.predict_proba(self.X_test)
        
        # Handle binary vs multiclass output
        if len(test_preds.shape) == 1:
            test_preds_class = (test_preds > 0.5).astype(int)
            test_probs = test_preds
        else:
            test_preds_class = np.argmax(test_preds, axis=1)
            test_probs = test_preds[:, 1]
        
        from sklearn.metrics import accuracy_score, log_loss
        accuracy = accuracy_score(self.y_test_outcome, test_preds_class)
        
        try:
            logloss = log_loss(self.y_test_outcome, test_preds)
            logger.info(f"Ensemble Test Log Loss: {logloss:.4f}")
        except:
            pass
        
        logger.info(f"Ensemble Test Accuracy: {accuracy:.4f}")
        
        # NEW: Threshold analysis for ensemble
        logger.info("\n" + "=" * 80)
        logger.info("ENSEMBLE THRESHOLD ANALYSIS")
        logger.info("=" * 80)
        
        ensemble_threshold_df = analyze_prediction_thresholds(self.y_test_outcome, test_probs)
        print_threshold_analysis(ensemble_threshold_df, logger)
        save_threshold_analysis(ensemble_threshold_df, output_dir='models/saved/ensemble_threshold_analysis.csv')
        
        # Store ensemble threshold analysis
        self.ensemble_threshold_analysis = ensemble_threshold_df
        
        return stacking
        
    def save_models(self, output_dir: str = 'src/models/saved'):
        """
        UPDATED: Save models including Poisson model
        """
        from pathlib import Path
        import joblib
        import pickle
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {output_dir}")
        
        # Save XGBoost outcome model
        if 'xgboost_outcome' in self.models:
            self.models['xgboost_outcome'].save_model(f"{output_dir}/xgboost_outcome")
            logger.info("  Saved XGBoost outcome model")
        
        # CHANGED: Save Poisson Multi-task Neural Network
        if 'poisson_multitask' in self.models:
            self.models['poisson_multitask'].save_checkpoint(f"{output_dir}/poisson_multitask.pt")
            logger.info("  Saved Poisson Multi-task Neural Network")
        
        # Save LightGBM model (if it exists)
        if 'lightgbm_outcome' in self.models:
            joblib.dump(self.models['lightgbm_outcome'], f"{output_dir}/lightgbm_outcome.pkl")
            logger.info("  Saved LightGBM model")
        
        # Save Stacking Ensemble
        if 'stacking_ensemble' in self.models:
            with open(f"{output_dir}/stacking_ensemble.pkl", 'wb') as f:
                pickle.dump(self.models['stacking_ensemble'], f)
            logger.info("  Saved Stacking Ensemble")
        
        # Save feature columns
        joblib.dump(self.feature_columns, f"{output_dir}/feature_columns.pkl")
        logger.info("  Saved feature columns")
        
        # Save threshold analysis if available
        if hasattr(self, 'threshold_analysis'):
            self.threshold_analysis.to_csv(f"{output_dir}/threshold_analysis.csv", index=False)
            logger.info("  Saved threshold analysis")
        
        # NEW: Save O/U analysis results if available
        if hasattr(self, 'score_analyzer'):
            logger.info("  Score analysis plots and edges already saved")
        
        # Save configuration
        with open(f"{output_dir}/model_config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        logger.info("  Saved model configuration")
        
        # Create model inventory file
        inventory = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'models_saved': list(self.models.keys()),
            'n_features': len(self.feature_columns),
            'test_size': len(self.X_test),
            'train_size': len(self.X_train),
            'uses_poisson': 'poisson_multitask' in self.models  # NEW: Track if using Poisson
        }
        
        with open(f"{output_dir}/model_inventory.yaml", 'w') as f:
            yaml.dump(inventory, f)
        logger.info("  Saved model inventory")
        
        logger.info(f"\nAll {len(self.models)} models saved successfully to {output_dir}/")
    
    def run_full_pipeline(self, connection_string: str):
        """
        UPDATED: Run complete training pipeline with Poisson scoring
        """
        logger.info("=" * 80)
        logger.info("STARTING NHL PREDICTION PIPELINE (WITH POISSON)")
        logger.info("=" * 80)
        
        # Load data
        self.load_data(connection_string, use_cache=False)
        
        # Engineer features
        self.engineer_features()
        
        # Prepare train/test split
        self.prepare_train_test_split()
        
        # Train models
        self.train_level_1_outcome_classifier()
        
        # CHANGED: Train Poisson score prediction instead of regular NN
        # self.train_level_2_score_prediction_poisson()
        
        # # NEW: Analyze score predictions and find O/U edges
        # self.analyze_score_predictions()
        
        # Train ensemble (for outcome predictions)
        self.train_ensemble()
        
        # Save all models
        self.save_models()
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE - POISSON MODEL TRAINED")
        logger.info("=" * 80)
        
        # NEW: Print summary of betting opportunities
        logger.info("\n" + "=" * 80)
        logger.info("BETTING OPPORTUNITY SUMMARY")
        logger.info("=" * 80)
        
        # Outcome betting
        if hasattr(self, 'threshold_analysis'):
            best_threshold = self.threshold_analysis.loc[
                self.threshold_analysis['high_conf_accuracy'].idxmax()
            ]
            logger.info(f"\nMoneyline Betting:")
            logger.info(f"  Best threshold: {best_threshold['threshold']:.2f}")
            logger.info(f"  Accuracy: {best_threshold['high_conf_accuracy']:.4f}")
            logger.info(f"  Coverage: {best_threshold['coverage_pct']:.1f}% of games")
        
        # O/U betting
        # if hasattr(self, 'score_analyzer'):
        #     edges_df = self.score_analyzer.find_betting_edges_improved()
        #     if not edges_df.empty:
        #         logger.info(f"\nOver/Under Betting:")
        #         logger.info(f"  Opportunities found: {len(edges_df)}")
        #         logger.info(f"  Average edge: {edges_df['edge'].mean():.3f}")
        #         logger.info(f"  Expected ROI: {edges_df['roi'].mean():.3f}")
                
        #         # Best edges by line
        #         for line in edges_df['line'].unique():
        #             line_edges = edges_df[edges_df['line'] == line]
        #             logger.info(f"  Line {line}: {len(line_edges)} bets, "
        #                       f"Win rate: {line_edges['hit'].mean():.3f}")
        
        logger.info("=" * 80)

def identify_leaking_features(df, feature_columns):
    """
    Identify features that might contain current game information
    and should be excluded from prediction features
    """

    leaking_patterns = [
        'won',           # Current game outcome
        'points',        # Current game points (unless rolling)
        'goal_differential',  # Current game differential (unless rolling)
        'goals_for',     # Current game goals (unless rolling)
        'goals_against', # Current game goals (unless rolling)
        'shots_for',     # Current game shots (unless rolling)
        'shots_against', # Current game shots (unless rolling)
        'xG_for',        # Current game xG (unless rolling)
        'xG_against',    # Current game xG (unless rolling)
        'shooting_percentage',  # ADD THIS - raw percentage (unless rolling)
        'save_percentage',      # ADD THIS - raw percentage (unless rolling)
        'xG_diff',             # ADD THIS - raw diff (unless rolling)
        'goalie_id',     # Not predictive
        'opponent_goalie_id',  # Not predictive
        'opponent_team_id',    # Already captured in matchup
    ]
    
    safe_patterns = [
        'rolling',       # Rolling averages are safe
        'ema',           # Exponential moving averages are safe
        'trend',         # Trend features are safe
        'streak',        # Streak features are safe
        'momentum',      # Momentum features are safe
        'consistency',   # Consistency metrics are safe
        'rest_days',     # Rest days are safe
        'is_home',       # Home/away indicator is safe
        'games_played',  # Games played count is safe
        'season_progress', # Season progress is safe
        'month',         # Month is safe
        'day_of_week',   # Day of week is safe
        'is_back_to_back',   # Back-to-back indicator is safe
        'is_well_rested',    # Well-rested indicator is safe
    ]
    
    # Check each feature
    leaking_features = []
    safe_features = []
    
    for col in feature_columns:
        # Check if it's a safe pattern
        is_safe = any(pattern in col for pattern in safe_patterns)
        
        if is_safe:
            safe_features.append(col)
        else:
            # Check if it matches a leaking pattern
            is_leaking = any(pattern in col for pattern in leaking_patterns)
            if is_leaking:
                leaking_features.append(col)
            else:
                # If unsure, include it but log warning
                safe_features.append(col)
                logger.warning(f"Feature included but verify: {col}")
    
    return safe_features, leaking_features

def detect_overtime_game(period_descriptor):
        """
        Detect if a game went to overtime/shootout
        
        Args:
            period_descriptor: Can be dict, string, or None
            
        Returns:
            bool: True if game went to OT/SO, False otherwise
        """
        if pd.isna(period_descriptor):
            return False
        
        try:
            # If it's already a dict
            if isinstance(period_descriptor, dict):
                period_type = period_descriptor.get('periodType', '')
                return period_type in ['OT', 'SO']
            
            # If it's a string representation of JSON
            elif isinstance(period_descriptor, str):
                # Try to parse as JSON
                try:
                    import json
                    period_data = json.loads(period_descriptor)
                    period_type = period_data.get('periodType', '')
                    return period_type in ['OT', 'SO']
                except json.JSONDecodeError:
                    # If not valid JSON, check if 'OT' or 'SO' in string
                    return 'OT' in period_descriptor or 'SO' in period_descriptor
            
            return False
        except Exception as e:
            logger.warning(f"Error detecting OT: {e}")
            return False

def analyze_prediction_thresholds(y_true, y_probs, thresholds=None):
    """
    Comprehensive threshold analysis for binary classification
    
    Args:
        y_true: True labels (0 or 1)
        y_probs: Predicted probabilities for class 1 (home win)
        thresholds: List of probability thresholds to analyze
        
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    results = []
    
    for threshold in thresholds:
        # Make predictions at this threshold
        y_pred = (y_probs >= threshold).astype(int)
        
        # Count predictions
        n_predictions = len(y_pred)
        n_home_predicted = (y_pred == 1).sum()
        n_away_predicted = (y_pred == 0).sum()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate coverage (% of predictions meeting threshold)
        high_conf_home = (y_probs >= threshold).sum()
        high_conf_away = (y_probs < (1 - threshold)).sum()
        total_high_conf = high_conf_home + high_conf_away
        coverage = (total_high_conf / n_predictions) * 100
        
        # Calculate accuracy on high confidence predictions only
        high_conf_mask = (y_probs >= threshold) | (y_probs < (1 - threshold))
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
        else:
            high_conf_acc = 0.0
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_total': n_predictions,
            'n_home_predicted': n_home_predicted,
            'n_away_predicted': n_away_predicted,
            'pct_home_predicted': (n_home_predicted / n_predictions) * 100,
            'high_confidence_count': total_high_conf,
            'coverage_pct': coverage,
            'high_conf_accuracy': high_conf_acc
        })
    
    return pd.DataFrame(results)


def print_threshold_analysis(threshold_df, logger):
    """
    Pretty print threshold analysis results
    
    Args:
        threshold_df: DataFrame from analyze_prediction_thresholds
        logger: Logger instance
    """
    logger.info("\n" + "=" * 100)
    logger.info("COMPREHENSIVE THRESHOLD ANALYSIS")
    logger.info("=" * 100)
    
    logger.info("\n{:<12} {:<10} {:<10} {:<10} {:<8} {:<12} {:<12} {:<12}".format(
        "Threshold", "Accuracy", "Precision", "Recall", "F1", "Coverage%", "HighConfAcc", "HomeWinPct"
    ))
    logger.info("-" * 100)
    
    for _, row in threshold_df.iterrows():
        logger.info("{:<12.2f} {:<10.4f} {:<10.4f} {:<10.4f} {:<8.4f} {:<12.1f} {:<12.4f} {:<12.1f}".format(
            row['threshold'],
            row['accuracy'],
            row['precision'],
            row['recall'],
            row['f1_score'],
            row['coverage_pct'],
            row['high_conf_accuracy'],
            row['pct_home_predicted']
        ))
    
    logger.info("-" * 100)
    logger.info("\nKey Insights:")
    
    # Find best threshold by accuracy
    best_acc_idx = threshold_df['accuracy'].idxmax()
    best_acc_row = threshold_df.loc[best_acc_idx]
    logger.info(f"  Best Accuracy: {best_acc_row['accuracy']:.4f} at threshold {best_acc_row['threshold']:.2f}")
    
    # Find threshold with good coverage and accuracy
    good_coverage = threshold_df[threshold_df['coverage_pct'] >= 50]
    if not good_coverage.empty:
        best_coverage_idx = good_coverage['high_conf_accuracy'].idxmax()
        best_coverage_row = good_coverage.loc[best_coverage_idx]
        logger.info(f"  Best High-Confidence Accuracy: {best_coverage_row['high_conf_accuracy']:.4f} at threshold {best_coverage_row['threshold']:.2f}")
        logger.info(f"    (covers {best_coverage_row['coverage_pct']:.1f}% of games)")
    
    # Recommended betting thresholds
    logger.info("\n  Recommended Betting Thresholds:")
    for threshold in [0.60, 0.65, 0.70]:
        row = threshold_df[threshold_df['threshold'] == threshold]
        if not row.empty:
            row = row.iloc[0]
            logger.info(f"    {threshold:.0%}: Accuracy={row['accuracy']:.4f}, Coverage={row['coverage_pct']:.1f}%, HighConfAcc={row['high_conf_accuracy']:.4f}")
    
    logger.info("=" * 100 + "\n")


def save_threshold_analysis(threshold_df, output_dir='src/models/saved'):
    """
    Save threshold analysis to CSV
    
    Args:
        threshold_df: DataFrame from analyze_prediction_thresholds
        output_dir: Directory to save the CSV
    """
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/threshold_analysis.csv"
    threshold_df.to_csv(output_path, index=False)
    logger.info(f"Threshold analysis saved to {output_path}")

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
    """Main execution function"""
    
    # Database connection string (update with your credentials)
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=nhlDB;"
        "Trusted_Connection=yes;"
    )
    
    # Initialize and run pipeline
    pipeline = NHLPredictionPipeline()
    pipeline.run_full_pipeline(connection_string)


if __name__ == "__main__":
    main()