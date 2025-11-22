"""
Feature Validation and Diagnostic Checks for NHL Predictor
Add these methods to your TodaysGamesPredictor class
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class FeatureValidator:
    """Comprehensive feature validation for prediction pipeline"""
    
    @staticmethod
    def validate_feature_alignment(
        prediction_features: pd.DataFrame,
        expected_features: List[str],
        game_info: str = "Current Game"
    ) -> Dict:
        """
        Comprehensive validation of feature alignment between prediction and training
        
        Args:
            prediction_features: DataFrame with features prepared for prediction
            expected_features: List of feature names from training (from feature_columns.pkl)
            game_info: String identifier for the game being validated
            
        Returns:
            Dict with validation results and diagnostics
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        pred_features_set = set(prediction_features.columns)
        expected_features_set = set(expected_features)
        
        # ============================================================
        # CHECK 1: Missing Features
        # ============================================================
        missing_features = expected_features_set - pred_features_set
        if missing_features:
            results['valid'] = False
            results['errors'].append(
                f"MISSING {len(missing_features)} features that model expects:"
            )
            for feat in sorted(list(missing_features))[:20]:  # Show first 20
                results['errors'].append(f"  - {feat}")
            if len(missing_features) > 20:
                results['errors'].append(f"  ... and {len(missing_features) - 20} more")
        
        # ============================================================
        # CHECK 2: Extra Features (not in training)
        # ============================================================
        extra_features = pred_features_set - expected_features_set
        if extra_features:
            results['warnings'].append(
                f"Found {len(extra_features)} extra features not in training set:"
            )
            for feat in sorted(list(extra_features))[:10]:
                results['warnings'].append(f"  - {feat}")
            if len(extra_features) > 10:
                results['warnings'].append(f"  ... and {len(extra_features) - 10} more")
        
        # ============================================================
        # CHECK 3: Feature Count Match
        # ============================================================
        if len(prediction_features.columns) != len(expected_features):
            results['errors'].append(
                f"Feature count mismatch: Prediction={len(prediction_features.columns)}, "
                f"Expected={len(expected_features)}"
            )
            results['valid'] = False
        else:
            results['info'].append(
                f"✅ Feature count matches: {len(expected_features)} features"
            )
        
        # ============================================================
        # CHECK 4: Feature Order Match
        # ============================================================
        if list(prediction_features.columns) != expected_features:
            results['warnings'].append(
                "Feature order differs from training (will be reordered)"
            )
        else:
            results['info'].append("✅ Feature order matches training")
        
        # ============================================================
        # CHECK 5: Data Type Validation
        # ============================================================
        non_numeric = []
        for col in prediction_features.columns:
            dtype = prediction_features[col].dtype
            if dtype not in ['int64', 'float64', 'int32', 'float32', 'bool', 'int8', 'int16']:
                non_numeric.append(f"{col} ({dtype})")
        
        if non_numeric:
            results['errors'].append(
                f"Found {len(non_numeric)} non-numeric features:"
            )
            for item in non_numeric[:10]:
                results['errors'].append(f"  - {item}")
            results['valid'] = False
        else:
            results['info'].append("✅ All features are numeric")
        
        # ============================================================
        # CHECK 6: NaN/Inf Values
        # ============================================================
        nan_counts = prediction_features.isnull().sum()
        features_with_nan = nan_counts[nan_counts > 0]
        
        if len(features_with_nan) > 0:
            results['errors'].append(
                f"Found NaN values in {len(features_with_nan)} features:"
            )
            for feat, count in features_with_nan.head(10).items():
                results['errors'].append(f"  - {feat}: {count} NaN values")
            results['valid'] = False
        else:
            results['info'].append("✅ No NaN values found")
        
        # Check for infinite values
        inf_features = []
        for col in prediction_features.select_dtypes(include=[np.number]).columns:
            if np.isinf(prediction_features[col]).any():
                inf_features.append(col)
        
        if inf_features:
            results['errors'].append(
                f"Found infinite values in {len(inf_features)} features:"
            )
            for feat in inf_features[:10]:
                results['errors'].append(f"  - {feat}")
            results['valid'] = False
        else:
            results['info'].append("✅ No infinite values found")
        
        # ============================================================
        # CHECK 7: Feature Value Ranges (sanity check)
        # ============================================================
        suspicious_features = []
        for col in prediction_features.columns:
            values = prediction_features[col].values[0]
            
            # Check for extreme values
            if abs(values) > 1e10:
                suspicious_features.append(f"{col}: {values:.2e} (extremely large)")
            
            # Check if all zeros (might indicate missing data)
            if values == 0 and 'rolling' in col.lower():
                suspicious_features.append(f"{col}: all zeros (check data availability)")
        
        if suspicious_features:
            results['warnings'].append(
                f"Suspicious feature values detected ({len(suspicious_features)}):"
            )
            for item in suspicious_features[:5]:
                results['warnings'].append(f"  - {item}")
        
        # ============================================================
        # Summary
        # ============================================================
        results['summary'] = {
            'game_info': game_info,
            'total_features': len(prediction_features.columns),
            'expected_features': len(expected_features),
            'missing_features': len(missing_features),
            'extra_features': len(extra_features),
            'nan_features': len(features_with_nan),
            'inf_features': len(inf_features),
            'validation_passed': results['valid']
        }
        
        return results
    
    @staticmethod
    def print_validation_report(validation_results: Dict):
        """Pretty print validation results"""
        
        print("\n" + "="*100)
        print("FEATURE VALIDATION REPORT")
        print("="*100)
        
        summary = validation_results['summary']
        print(f"\nGame: {summary['game_info']}")
        print(f"Total Features: {summary['total_features']}")
        print(f"Expected Features: {summary['expected_features']}")
        print(f"Validation Status: {'✅ PASSED' if summary['validation_passed'] else '❌ FAILED'}")
        
        if validation_results['info']:
            print("\n" + "-"*100)
            print("ℹ️  INFORMATION:")
            print("-"*100)
            for info in validation_results['info']:
                print(f"  {info}")
        
        if validation_results['warnings']:
            print("\n" + "-"*100)
            print("⚠️  WARNINGS:")
            print("-"*100)
            for warning in validation_results['warnings']:
                print(f"  {warning}")
        
        if validation_results['errors']:
            print("\n" + "-"*100)
            print("❌ ERRORS:")
            print("-"*100)
            for error in validation_results['errors']:
                print(f"  {error}")
        
        print("\n" + "="*100)
        
        if not summary['validation_passed']:
            print("\n⚠️  VALIDATION FAILED - FIX ERRORS BEFORE PROCEEDING")
            print("="*100)
    
    @staticmethod
    def compare_feature_distributions(
        current_features: pd.DataFrame,
        feature_name: str,
        expected_range: Tuple[float, float] = None
    ) -> Dict:
        """
        Compare current feature values against expected ranges
        
        Args:
            current_features: DataFrame with current features
            feature_name: Name of feature to check
            expected_range: Optional tuple of (min, max) expected values
            
        Returns:
            Dict with comparison results
        """
        if feature_name not in current_features.columns:
            return {
                'valid': False,
                'error': f"Feature '{feature_name}' not found"
            }
        
        value = current_features[feature_name].values[0]
        
        result = {
            'feature': feature_name,
            'value': value,
            'valid': True
        }
        
        if expected_range:
            min_val, max_val = expected_range
            if value < min_val or value > max_val:
                result['valid'] = False
                result['warning'] = (
                    f"Value {value:.4f} outside expected range "
                    f"[{min_val:.4f}, {max_val:.4f}]"
                )
        
        return result