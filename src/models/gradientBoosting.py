import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.model_selection import cross_val_score
import joblib

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost model with GPU acceleration for NHL game prediction"""
    
    def __init__(self, task: str = 'classification', params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model
        
        Args:
            task: 'classification', 'regression', or 'multiclass'
            params: Model hyperparameters
        """
        self.task = task
        self.model = None
        self.feature_names = None
        self.best_iteration = None
        
        # Default parameters optimized for GPU
        self.params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'predictor': 'gpu_predictor',
            'max_depth': 8,
            'learning_rate': 0.03,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with user params
        if params:
            self.params.update(params)
        
        # Set objective based on task
        if task == 'classification':
            self.params['objective'] = 'binary:logistic'
            self.params['eval_metric'] = ['logloss', 'auc']
        elif task == 'multiclass':
            self.params['objective'] = 'multi:softprob'
            self.params['eval_metric'] = ['mlogloss']
        elif task == 'regression':
            self.params['objective'] = 'reg:squarederror'
            self.params['eval_metric'] = ['rmse', 'mae']
        
        logger.info(f"Initialized XGBoost {task} model with GPU acceleration")
    
    def train(self, 
              X_train: pd.DataFrame,
              y_train: np.ndarray,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 50,
              verbose: int = 50) -> Dict[str, Any]:
        """
        Train XGBoost model with early stopping
        
        Returns:
            Dictionary with training history and metrics
        """
        self.feature_names = X_train.columns.tolist()
        
        # Create DMatrix for efficient GPU processing
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        eval_list = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            eval_list.append((dval, 'validation'))
        
        # Training with callbacks
        evals_result = {}
        
        logger.info("Training XGBoost model on GPU...")
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 1000),
            evals=eval_list,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose
        )
        
        self.best_iteration = self.model.best_iteration
        logger.info(f"Training completed. Best iteration: {self.best_iteration}")
        
        return evals_result
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        
        if self.task == 'multiclass':
            # Returns probability for each class
            return self.model.predict(dtest)
        else:
            return self.model.predict(dtest)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if self.task == 'regression':
            raise ValueError("predict_proba not available for regression tasks")
        
        return self.predict(X)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            importance_type: 'gain', 'weight', 'cover', or 'total_gain'
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.get_score(importance_type=importance_type)
        
        # Create dataframe
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for interpretability"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest, pred_contribs=True)
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save XGBoost model
        self.model.save_model(f"{filepath}.json")
        
        # Save additional attributes
        metadata = {
            'feature_names': self.feature_names,
            'params': self.params,
            'task': self.task,
            'best_iteration': self.best_iteration
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(f"{filepath}.json")
        
        # Load metadata
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        self.feature_names = metadata['feature_names']
        self.params = metadata['params']
        self.task = metadata['task']
        self.best_iteration = metadata['best_iteration']
        
        logger.info(f"Model loaded from {filepath}")
    
    def cross_validate(self,
                      X: pd.DataFrame,
                      y: np.ndarray,
                      cv: int = 5,
                      scoring: str = 'neg_log_loss') -> Dict[str, float]:
        """
        Perform cross-validation
        
        Returns:
            Dictionary with mean and std of scores
        """
        # Create sklearn-compatible model
        if self.task == 'classification':
            model = xgb.XGBClassifier(**self.params)
        elif self.task == 'multiclass':
            model = xgb.XGBClassifier(**self.params)
        else:
            model = xgb.XGBRegressor(**self.params)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores
        }


class XGBoostEnsemble:
    """Ensemble of XGBoost models for robust predictions"""
    
    def __init__(self, n_models: int = 5, task: str = 'classification', base_params: Optional[Dict] = None):
        self.n_models = n_models
        self.task = task
        self.models = []
        self.base_params = base_params or {}
        
    def train(self,
             X_train: pd.DataFrame,
             y_train: np.ndarray,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[np.ndarray] = None):
        """Train ensemble with different random seeds"""
        
        for i in range(self.n_models):
            logger.info(f"Training ensemble model {i+1}/{self.n_models}")
            
            # Different random seed for each model
            params = self.base_params.copy()
            params['random_state'] = 42 + i
            
            model = XGBoostModel(task=self.task, params=params)
            model.train(X_train, y_train, X_val, y_val)
            
            self.models.append(model)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average predictions from all models"""
        predictions = np.array([model.predict(X) for model in self.models])
        return predictions.mean(axis=0)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with uncertainty estimates
        
        Returns:
            mean_predictions, std_predictions
        """
        predictions = np.array([model.predict(X) for model in self.models])
        return predictions.mean(axis=0), predictions.std(axis=0)
    
    def save_ensemble(self, base_filepath: str):
        """Save all models in ensemble"""
        for i, model in enumerate(self.models):
            model.save_model(f"{base_filepath}_model_{i}")
        
        logger.info(f"Ensemble saved to {base_filepath}")
    
    def load_ensemble(self, base_filepath: str):
        """Load all models in ensemble"""
        self.models = []
        for i in range(self.n_models):
            model = XGBoostModel(task=self.task)
            model.load_model(f"{base_filepath}_model_{i}")
            self.models.append(model)
        
        logger.info(f"Ensemble loaded from {base_filepath}")