import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)

class StackingEnsemble:
    """
    Stacked ensemble that combines multiple base models with a meta-learner
    Uses out-of-fold predictions to avoid overfitting
    """
    
    def __init__(self,
                 base_models: List[Any],
                 meta_learner: str = 'logistic',
                 cv_folds: int = 5,
                 use_probas: bool = True,
                 task: str = 'classification'):
        """
        Args:
            base_models: List of trained base models
            meta_learner: 'logistic', 'lightgbm', or custom model
            cv_folds: Number of CV folds for generating meta-features
            use_probas: Use predicted probabilities (vs class predictions)
            task: 'classification', 'multiclass', or 'regression'
        """
        self.base_models = base_models
        self.meta_learner_type = meta_learner
        self.cv_folds = cv_folds
        self.use_probas = use_probas
        self.task = task
        self.meta_learner = None
        self.scaler = StandardScaler()
        
    def _create_meta_learner(self, n_features: int):
        """Create meta-learner model"""
        if self.meta_learner_type == 'logistic':
            if self.task == 'multiclass':
                self.meta_learner = LogisticRegression(
                    multi_class='multinomial',
                    max_iter=1000,
                    random_state=42
                )
            elif self.task == 'classification':
                self.meta_learner = LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            else:
                from sklearn.linear_model import Ridge
                self.meta_learner = Ridge(alpha=1.0)
                
        elif self.meta_learner_type == 'lightgbm':
            if self.task == 'multiclass':
                self.meta_learner = lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    device='gpu'
                )
            elif self.task == 'classification':
                self.meta_learner = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    device='gpu'
                )
            else:
                self.meta_learner = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    device='gpu'
                )
        
        logger.info(f"Created meta-learner: {self.meta_learner_type}")
    
    def generate_meta_features(self,
                              X: pd.DataFrame,
                              y: np.ndarray,
                              fit_base: bool = False) -> np.ndarray:
        """
        Generate meta-features using out-of-fold predictions
        
        Args:
            X: Training features
            y: Training targets
            fit_base: Whether to fit base models (True for training, False for inference)
        
        Returns:
            Meta-features array
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        
        if self.use_probas and self.task == 'multiclass':
            n_classes = 3
            meta_features = np.zeros((n_samples, n_models * n_classes))
        elif self.use_probas and self.task == 'classification':
            meta_features = np.zeros((n_samples, n_models))
        else:
            meta_features = np.zeros((n_samples, n_models))
        
        if fit_base:
            # Use K-Fold CV to generate out-of-fold predictions
            if self.task in ['classification', 'multiclass']:
                kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            else:
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            logger.info(f"Generating out-of-fold predictions with {self.cv_folds} folds")
            
            for model_idx, model in enumerate(self.base_models):
                logger.info(f"Processing base model {model_idx + 1}/{n_models}")
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    
                    # Train model on fold
                    if hasattr(model, 'train'):
                        model.train(X_train_fold, y_train_fold)
                    else:
                        model.fit(X_train_fold, y_train_fold)
                    
                    # Get predictions for validation fold
                    if self.use_probas:
                        if hasattr(model, 'predict_proba'):
                            preds = model.predict_proba(X_val_fold)
                        else:
                            preds = model.predict(X_val_fold)
                        
                        if self.task == 'multiclass':
                            # Store all class probabilities
                            meta_features[val_idx, model_idx*n_classes:(model_idx+1)*n_classes] = preds
                        else:
                            # Binary classification - use probability of positive class
                            if len(preds.shape) > 1:
                                meta_features[val_idx, model_idx] = preds[:, 1]
                            else:
                                meta_features[val_idx, model_idx] = preds
                    else:
                        preds = model.predict(X_val_fold)
                        meta_features[val_idx, model_idx] = preds
            
            # After CV, retrain all models on full dataset
            logger.info("Retraining all base models on full dataset")
            for model_idx, model in enumerate(self.base_models):
                if hasattr(model, 'train'):
                    model.train(X, y)
                else:
                    model.fit(X, y)
        
        else:
            # Inference mode - use trained models directly
            for model_idx, model in enumerate(self.base_models):
                if self.use_probas:
                    if hasattr(model, 'predict_proba'):
                        preds = model.predict_proba(X)
                    else:
                        preds = model.predict(X)
                    
                    if self.task == 'multiclass':
                        meta_features[:, model_idx*n_classes:(model_idx+1)*n_classes] = preds
                    else:
                        if len(preds.shape) > 1:
                            meta_features[:, model_idx] = preds[:, 1]
                        else:
                            meta_features[:, model_idx] = preds
                else:
                    preds = model.predict(X)
                    meta_features[:, model_idx] = preds
        
        return meta_features
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit the stacking ensemble
        
        Args:
            X: Training features
            y: Training targets
        """
        logger.info("Training stacking ensemble...")
        
        # Generate meta-features using out-of-fold predictions
        meta_features = self.generate_meta_features(X, y, fit_base=True)
        
        # Standardize meta-features
        meta_features = self.scaler.fit_transform(meta_features)
        
        # Create and train meta-learner
        self._create_meta_learner(meta_features.shape[1])
        
        logger.info("Training meta-learner...")
        self.meta_learner.fit(meta_features, y)
        
        logger.info("Stacking ensemble training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble"""
        # Generate meta-features from base models
        meta_features = self.generate_meta_features(X, y=None, fit_base=False)
        
        # Standardize
        meta_features = self.scaler.transform(meta_features)
        
        # Meta-learner prediction
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.task == 'regression':
            raise ValueError("predict_proba not available for regression")
        
        meta_features = self.generate_meta_features(X, y=None, fit_base=False)
        meta_features = self.scaler.transform(meta_features)
        
        return self.meta_learner.predict_proba(meta_features)
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get importance/weights of base models in the ensemble
        Only works for linear meta-learners
        """
        if not isinstance(self.meta_learner, LogisticRegression):
            logger.warning("Model weights only available for logistic regression meta-learner")
            return {}
        
        weights = {}
        coefs = self.meta_learner.coef_[0] if len(self.meta_learner.coef_.shape) == 2 else self.meta_learner.coef_
        
        n_models = len(self.base_models)
        if self.task == 'multiclass':
            n_classes = 3
            for i in range(n_models):
                avg_weight = np.abs(coefs[i*n_classes:(i+1)*n_classes]).mean()
                weights[f'model_{i}'] = avg_weight
        else:
            for i in range(n_models):
                weights[f'model_{i}'] = abs(coefs[i])
        
        return weights


class BlendingEnsemble:
    """
    Simpler blending ensemble that trains base models on train set
    and meta-learner on validation set
    """
    
    def __init__(self,
                 base_models: List[Any],
                 meta_learner: str = 'logistic',
                 task: str = 'classification'):
        self.base_models = base_models
        self.meta_learner_type = meta_learner
        self.task = task
        self.meta_learner = None
        self.scaler = StandardScaler()
    
    def fit(self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame,
            y_val: np.ndarray):
        """
        Fit blending ensemble
        
        Args:
            X_train: Training features for base models
            y_train: Training targets for base models
            X_val: Validation features for meta-learner
            y_val: Validation targets for meta-learner
        """
        logger.info("Training blending ensemble...")
        
        # Train base models on training set
        for idx, model in enumerate(self.base_models):
            logger.info(f"Training base model {idx + 1}/{len(self.base_models)}")
            if hasattr(model, 'train'):
                model.train(X_train, y_train)
            else:
                model.fit(X_train, y_train)
        
        # Generate predictions on validation set
        meta_features = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X_val)
                if len(preds.shape) > 1 and self.task != 'multiclass':
                    preds = preds[:, 1]
            else:
                preds = model.predict(X_val)
            meta_features.append(preds)
        
        meta_features = np.column_stack(meta_features)
        meta_features = self.scaler.fit_transform(meta_features)
        
        # Train meta-learner
        if self.meta_learner_type == 'logistic':
            if self.task == 'multiclass':
                self.meta_learner = LogisticRegression(multi_class='multinomial', max_iter=1000)
            else:
                self.meta_learner = LogisticRegression(max_iter=1000)
        elif self.meta_learner_type == 'lightgbm':
            if self.task == 'multiclass':
                self.meta_learner = lgb.LGBMClassifier(objective='multiclass', num_class=3)
            else:
                self.meta_learner = lgb.LGBMClassifier()
        
        self.meta_learner.fit(meta_features, y_val)
        logger.info("Blending ensemble training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        meta_features = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
                if len(preds.shape) > 1 and self.task != 'multiclass':
                    preds = preds[:, 1]
            else:
                preds = model.predict(X)
            meta_features.append(preds)
        
        meta_features = np.column_stack(meta_features)
        meta_features = self.scaler.transform(meta_features)
        
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        meta_features = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
                if len(preds.shape) > 1 and self.task != 'multiclass':
                    preds = preds[:, 1]
            else:
                preds = model.predict(X)
            meta_features.append(preds)
        
        meta_features = np.column_stack(meta_features)
        meta_features = self.scaler.transform(meta_features)
        
        return self.meta_learner.predict_proba(meta_features)