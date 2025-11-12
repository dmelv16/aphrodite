"""
GPU Utilities for optimizing training and inference
"""

import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, any]:
    """
    Check GPU availability and specifications
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'devices': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory / 1e9,  # GB
                'memory_allocated': torch.cuda.memory_allocated(i) / 1e9,
                'memory_cached': torch.cuda.memory_reserved(i) / 1e9
            }
            info['devices'].append(device_info)
            
            logger.info(f"GPU {i}: {device_info['name']}")
            logger.info(f"  Total Memory: {device_info['memory_total']:.2f} GB")
    else:
        logger.warning("No GPU available. Models will run on CPU.")
    
    return info


def get_optimal_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get optimal device for training
    
    Args:
        device_id: Specific GPU device ID (None for auto-select)
    
    Returns:
        torch.device
    """
    if not torch.cuda.is_available():
        logger.info("Using CPU")
        return torch.device('cpu')
    
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        # Auto-select GPU with most free memory
        best_device = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            free_memory = (
                torch.cuda.get_device_properties(i).total_memory - 
                torch.cuda.memory_allocated(i)
            )
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_device = i
        
        device = torch.device(f'cuda:{best_device}')
        logger.info(f"Auto-selected GPU {best_device}: {torch.cuda.get_device_name(best_device)}")
    
    return device


def clear_gpu_memory():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def set_gpu_memory_growth(enable: bool = True):
    """
    Enable/disable GPU memory growth (for TensorFlow if needed)
    
    Note: PyTorch handles memory automatically, but this is useful
    for mixed PyTorch/TensorFlow environments
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, enable)
            logger.info(f"TensorFlow GPU memory growth: {enable}")
    except ImportError:
        pass  # TensorFlow not installed


def optimize_cudnn():
    """Optimize cuDNN settings for performance"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info("cuDNN optimization enabled")


def get_gpu_memory_stats(device_id: int = 0) -> Dict[str, float]:
    """
    Get current GPU memory statistics
    
    Args:
        device_id: GPU device ID
    
    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    torch.cuda.synchronize(device_id)
    
    stats = {
        'allocated': torch.cuda.memory_allocated(device_id) / 1e9,
        'reserved': torch.cuda.memory_reserved(device_id) / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated(device_id) / 1e9,
        'max_reserved': torch.cuda.max_memory_reserved(device_id) / 1e9,
        'total': torch.cuda.get_device_properties(device_id).total_memory / 1e9
    }
    
    stats['free'] = stats['total'] - stats['reserved']
    stats['utilization'] = (stats['reserved'] / stats['total']) * 100
    
    return stats


def print_memory_summary(device_id: int = 0):
    """Print detailed memory summary"""
    if not torch.cuda.is_available():
        logger.info("No GPU available")
        return
    
    stats = get_gpu_memory_stats(device_id)
    
    logger.info(f"\nGPU {device_id} Memory Summary:")
    logger.info(f"  Total:         {stats['total']:.2f} GB")
    logger.info(f"  Allocated:     {stats['allocated']:.2f} GB")
    logger.info(f"  Reserved:      {stats['reserved']:.2f} GB")
    logger.info(f"  Free:          {stats['free']:.2f} GB")
    logger.info(f"  Utilization:   {stats['utilization']:.1f}%")
    logger.info(f"  Max Allocated: {stats['max_allocated']:.2f} GB")


def configure_mixed_precision() -> bool:
    """
    Check if mixed precision training is available
    
    Returns:
        True if supported, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    # Check if GPU supports mixed precision (compute capability >= 7.0)
    capability = torch.cuda.get_device_capability()
    supports_amp = capability[0] >= 7
    
    if supports_amp:
        logger.info(f"Mixed precision (AMP) supported (Compute Capability {capability[0]}.{capability[1]})")
    else:
        logger.warning(f"Mixed precision not supported (Compute Capability {capability[0]}.{capability[1]} < 7.0)")
    
    return supports_amp


class GPUMonitor:
    """Context manager for monitoring GPU usage during training"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.start_stats = None
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            self.start_stats = get_gpu_memory_stats(self.device_id)
            logger.info(f"GPU monitoring started for device {self.device_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_stats = get_gpu_memory_stats(self.device_id)
            
            logger.info(f"\nGPU Usage Summary:")
            logger.info(f"  Peak Memory: {end_stats['max_allocated']:.2f} GB")
            logger.info(f"  Final Memory: {end_stats['allocated']:.2f} GB")
            
            if self.start_stats:
                delta = end_stats['allocated'] - self.start_stats['allocated']
                logger.info(f"  Memory Change: {delta:+.2f} GB")


def setup_gpu_environment(device_id: Optional[int] = None, 
                         enable_benchmark: bool = True,
                         enable_amp: bool = True) -> torch.device:
    """
    Complete GPU environment setup
    
    Args:
        device_id: Specific GPU device ID (None for auto)
        enable_benchmark: Enable cuDNN benchmarking
        enable_amp: Check for mixed precision support
    
    Returns:
        Selected torch.device
    """
    logger.info("="*80)
    logger.info("GPU ENVIRONMENT SETUP")
    logger.info("="*80)
    
    # Check availability
    gpu_info = check_gpu_availability()
    
    # Get device
    device = get_optimal_device(device_id)
    
    # Optimize settings
    if enable_benchmark and torch.cuda.is_available():
        optimize_cudnn()
    
    # Check AMP support
    if enable_amp and torch.cuda.is_available():
        configure_mixed_precision()
    
    # Print memory stats
    if torch.cuda.is_available():
        print_memory_summary(device.index if device.type == 'cuda' else 0)
    
    logger.info("="*80)
    
    return device


# XGBoost-specific GPU utilities
def get_xgboost_gpu_params() -> Dict[str, any]:
    """Get optimal XGBoost GPU parameters"""
    if not torch.cuda.is_available():
        return {'tree_method': 'hist'}
    
    return {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'predictor': 'gpu_predictor',
    }


# LightGBM-specific GPU utilities  
def get_lightgbm_gpu_params() -> Dict[str, any]:
    """Get optimal LightGBM GPU parameters"""
    if not torch.cuda.is_available():
        return {'device': 'cpu'}
    
    return {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    }