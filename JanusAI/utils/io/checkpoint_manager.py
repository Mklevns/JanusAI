"""
Checkpoint Manager for Janus Framework

This module provides a dedicated CheckpointManager class responsible for robustly
managing model and training state checkpoints. It handles saving and loading of
PyTorch models, optimizers, training progress, and associated metadata with
version control and cleanup capabilities.

Key Features:
- Reliable persistence with atomic saves
- Version control with multiple checkpoint management
- Metadata tracking (epoch, metrics, hyperparameters)
- Experiment organization with ID-based subdirectories
- Automatic cleanup of old checkpoints
- CPU/GPU device handling
- Comprehensive error handling and recovery
"""

import os
import shutil
import glob
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import tempfile
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointError(Exception):
    """Custom exception for checkpoint operations."""
    pass


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint file is corrupted or invalid."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a requested checkpoint cannot be found."""
    pass


class CheckpointManager:
    """
    Manages checkpoint saving and loading for training recovery.
    
    Provides reliable persistence, version control, and metadata management
    for PyTorch models and training states with support for experiment organization.
    """

    def __init__(
        self, 
        checkpoint_dir: Union[str, Path], 
        experiment_id: Optional[str] = None,
        max_checkpoints: int = 5,
        auto_cleanup: bool = True
    ):
        """
        Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir: Base directory where all checkpoints will be stored
            experiment_id: Optional identifier for current experiment (creates subdirectory)
            max_checkpoints: Maximum number of recent checkpoints to keep (default: 5)
            auto_cleanup: Whether to automatically clean old checkpoints (default: True)
        """
        self.base_checkpoint_dir = Path(checkpoint_dir)
        
        # Create experiment-specific subdirectory if experiment_id provided
        if experiment_id:
            self.checkpoint_dir = self.base_checkpoint_dir / experiment_id
            self.experiment_id = experiment_id
        else:
            # Use timestamped default directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = self.base_checkpoint_dir / f"default_run_{timestamp}"
            self.experiment_id = f"default_run_{timestamp}"
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        
        # Metadata file for tracking checkpoints
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Device management for loading checkpoints
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        prefix: str = "ckpt",
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Save current state of model and training.
        
        Args:
            model: PyTorch model to save
            optimizer: PyTorch optimizer to save
            epoch: Current training epoch/iteration
            metrics: Dictionary of training metrics (loss, accuracy, etc.)
            prefix: Prefix for checkpoint filename
            extra_state: Additional state to save (schedulers, custom objects)
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Full path to the saved checkpoint file
            
        Raises:
            CheckpointError: If saving fails
        """
        try:
            # Generate checkpoint filename with metadata
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            # Include key metric in filename for easy identification
            main_metric = self._get_main_metric(metrics)
            metric_str = f"_{main_metric:.4f}" if main_metric is not None else ""
            
            filename = f"{prefix}_epoch_{epoch:04d}{metric_str}_{timestamp}.pt"
            checkpoint_path = self.checkpoint_dir / filename
            
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'timestamp': timestamp,
                'experiment_id': self.experiment_id,
                'model_class': model.__class__.__name__,
                'optimizer_class': optimizer.__class__.__name__,
            }
            
            # Add extra state if provided
            if extra_state:
                checkpoint_data['extra_state'] = extra_state
            
            # Add model architecture info for validation during loading
            checkpoint_data['model_info'] = {
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            # Atomic save operation to prevent corruption
            temp_path = checkpoint_path.with_suffix('.tmp')
            
            try:
                # Save to temporary file first
                torch.save(checkpoint_data, temp_path)
                
                # Verify the saved file by attempting to load it
                self._verify_checkpoint(temp_path)
                
                # Atomically move to final location
                temp_path.rename(checkpoint_path)
                
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                
            except Exception as e:
                # Clean up temporary file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                raise CheckpointError(f"Failed to save checkpoint atomically: {e}")
            
            # Update metadata
            self._update_metadata(filename, epoch, metrics, is_best)
            
            # Save best checkpoint separately if requested
            if is_best:
                best_path = self.checkpoint_dir / "checkpoint_best.pt"
                shutil.copy2(checkpoint_path, best_path)
                self.logger.info(f"Saved best checkpoint to {best_path}")
            
            # Automatic cleanup if enabled
            if self.auto_cleanup:
                self.clean_old_checkpoints(
                    keep_n=self.max_checkpoints, 
                    keep_best=True
                )
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
            raise CheckpointError(f"Failed to save checkpoint: {e}")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        latest: bool = True,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load a saved checkpoint into model and optimizer.
        
        Args:
            model: PyTorch model to load state into
            optimizer: Optional PyTorch optimizer to load state into
            checkpoint_path: Explicit path to checkpoint file (overrides latest)
            latest: If True and checkpoint_path is None, loads most recent checkpoint
            device: Target device for loading (defaults to manager's device)
            strict: Whether to strictly enforce state dict loading
            
        Returns:
            Dictionary containing loaded metadata (epoch, metrics, etc.)
            
        Raises:
            CheckpointNotFoundError: If checkpoint file not found
            CheckpointCorruptionError: If checkpoint file is corrupted
            CheckpointError: If loading fails for other reasons
        """
        try:
            # Determine which checkpoint to load
            if checkpoint_path:
                load_path = Path(checkpoint_path)
                if not load_path.is_absolute():
                    load_path = self.checkpoint_dir / checkpoint_path
            elif latest:
                load_path = self.get_latest_checkpoint_path()
                if load_path is None:
                    raise CheckpointNotFoundError(
                        f"No checkpoints found in {self.checkpoint_dir}"
                    )
                load_path = Path(load_path)
            else:
                raise CheckpointError("Must specify checkpoint_path or set latest=True")
            
            if not load_path.exists():
                raise CheckpointNotFoundError(f"Checkpoint file not found: {load_path}")
            
            # Determine device for loading
            target_device = device or self.device
            
            try:
                # Load checkpoint with appropriate device mapping
                if target_device.type == 'cpu':
                    checkpoint_data = torch.load(load_path, map_location='cpu')
                else:
                    checkpoint_data = torch.load(load_path, map_location=target_device)
                    
            except Exception as e:
                raise CheckpointCorruptionError(f"Failed to load checkpoint file: {e}")
            
            # Validate checkpoint structure
            self._validate_checkpoint_data(checkpoint_data)
            
            # Load model state
            try:
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
                self.logger.info("Loaded model state successfully")
            except Exception as e:
                if strict:
                    raise CheckpointError(f"Failed to load model state: {e}")
                else:
                    self.logger.warning(f"Partial model state loading: {e}")
            
            # Load optimizer state if provided
            if optimizer and 'optimizer_state_dict' in checkpoint_data:
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    self.logger.info("Loaded optimizer state successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to load optimizer state: {e}")
            
            # Extract metadata for return
            metadata = {
                'epoch': checkpoint_data.get('epoch', 0),
                'metrics': checkpoint_data.get('metrics', {}),
                'timestamp': checkpoint_data.get('timestamp'),
                'experiment_id': checkpoint_data.get('experiment_id'),
                'extra_state': checkpoint_data.get('extra_state', {})
            }
            
            self.logger.info(f"Successfully loaded checkpoint from {load_path}")
            
            return metadata
            
        except (CheckpointNotFoundError, CheckpointCorruptionError):
            raise
        except Exception as e:
            raise CheckpointError(f"Unexpected error during checkpoint loading: {e}")

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """
        Find the path to the most recently created checkpoint.
        
        Returns:
            Path string to latest checkpoint, or None if no checkpoints found
        """
        try:
            checkpoints = self.metadata.get('checkpoints', [])
            
            if not checkpoints:
                # Fallback: scan directory for checkpoint files
                checkpoint_files = list(self.checkpoint_dir.glob("ckpt_*.pt"))
                if not checkpoint_files:
                    return None
                
                # Sort by modification time
                latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
                return str(latest_file)
            
            # Use metadata to find latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: x['epoch'])
            return str(self.checkpoint_dir / latest_checkpoint['filename'])
            
        except Exception as e:
            self.logger.warning(f"Error finding latest checkpoint: {e}")
            return None

    def get_best_checkpoint_path(self, metric_name: str = "loss", minimize: bool = True) -> Optional[str]:
        """
        Find the path to the best checkpoint based on a specific metric.
        
        Args:
            metric_name: Name of the metric to optimize
            minimize: If True, lower values are better; if False, higher values are better
            
        Returns:
            Path string to best checkpoint, or None if not found
        """
        try:
            # Check for dedicated best checkpoint file
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            if best_path.exists():
                return str(best_path)
            
            # Find best based on metadata
            checkpoints = self.metadata.get('checkpoints', [])
            valid_checkpoints = [
                cp for cp in checkpoints 
                if metric_name in cp.get('metrics', {})
            ]
            
            if not valid_checkpoints:
                return None
            
            # Find checkpoint with best metric
            if minimize:
                best_checkpoint = min(
                    valid_checkpoints,
                    key=lambda x: x['metrics'][metric_name]
                )
            else:
                best_checkpoint = max(
                    valid_checkpoints,
                    key=lambda x: x['metrics'][metric_name]
                )
            
            return str(self.checkpoint_dir / best_checkpoint['filename'])
            
        except Exception as e:
            self.logger.warning(f"Error finding best checkpoint: {e}")
            return None

    def clean_old_checkpoints(
        self, 
        keep_n: int = 3, 
        keep_best: bool = True,
        metric_name: str = "loss", 
        minimize_metric: bool = True
    ) -> None:
        """
        Manage storage by deleting older or less performant checkpoints.
        
        Args:
            keep_n: Number of recent checkpoints to keep
            keep_best: Whether to preserve the best checkpoint regardless of age
            metric_name: Metric to use for determining "best" checkpoint
            minimize_metric: If True, lower metric values are better
        """
        try:
            checkpoints = self.metadata.get('checkpoints', [])
            
            if len(checkpoints) <= keep_n:
                return
            
            # Sort checkpoints by epoch (most recent first)
            sorted_checkpoints = sorted(checkpoints, key=lambda x: x['epoch'], reverse=True)
            
            # Identify checkpoints to keep
            keep_checkpoints = set()
            
            # Always keep the most recent ones
            for cp in sorted_checkpoints[:keep_n]:
                keep_checkpoints.add(cp['filename'])
            
            # Keep the best checkpoint if requested
            if keep_best:
                valid_checkpoints = [
                    cp for cp in checkpoints 
                    if metric_name in cp.get('metrics', {})
                ]
                
                if valid_checkpoints:
                    if minimize_metric:
                        best_cp = min(valid_checkpoints, key=lambda x: x['metrics'][metric_name])
                    else:
                        best_cp = max(valid_checkpoints, key=lambda x: x['metrics'][metric_name])
                    
                    keep_checkpoints.add(best_cp['filename'])
            
            # Remove checkpoints not in keep set
            removed_count = 0
            updated_checkpoints = []
            
            for cp in checkpoints:
                if cp['filename'] in keep_checkpoints:
                    updated_checkpoints.append(cp)
                else:
                    # Remove the file
                    checkpoint_path = self.checkpoint_dir / cp['filename']
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        removed_count += 1
                        self.logger.debug(f"Removed old checkpoint: {cp['filename']}")
            
            # Update metadata
            self.metadata['checkpoints'] = updated_checkpoints
            self._save_metadata()
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old checkpoints")
                
        except Exception as e:
            self.logger.error(f"Error during checkpoint cleanup: {e}")

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their metadata.
        
        Returns:
            List of dictionaries containing checkpoint information
        """
        return self.metadata.get('checkpoints', []).copy()

    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific checkpoint without loading it.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary with checkpoint information or None if not accessible
        """
        try:
            path = Path(checkpoint_path)
            if not path.is_absolute():
                path = self.checkpoint_dir / path
            
            if not path.exists():
                return None
            
            # Load only metadata (lightweight operation)
            checkpoint_data = torch.load(path, map_location='cpu')
            
            return {
                'filename': path.name,
                'size_mb': path.stat().st_size / (1024 * 1024),
                'epoch': checkpoint_data.get('epoch'),
                'metrics': checkpoint_data.get('metrics', {}),
                'timestamp': checkpoint_data.get('timestamp'),
                'model_class': checkpoint_data.get('model_class'),
                'model_info': checkpoint_data.get('model_info', {}),
                'experiment_id': checkpoint_data.get('experiment_id')
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get checkpoint info: {e}")
            return None

    def export_checkpoint(self, checkpoint_path: str, export_dir: Union[str, Path]) -> str:
        """
        Export a checkpoint to a different directory for sharing or backup.
        
        Args:
            checkpoint_path: Path to checkpoint to export
            export_dir: Directory to export to
            
        Returns:
            Path to exported checkpoint
        """
        try:
            source_path = Path(checkpoint_path)
            if not source_path.is_absolute():
                source_path = self.checkpoint_dir / source_path
            
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            export_path = export_dir / source_path.name
            shutil.copy2(source_path, export_path)
            
            self.logger.info(f"Exported checkpoint to {export_path}")
            return str(export_path)
            
        except Exception as e:
            raise CheckpointError(f"Failed to export checkpoint: {e}")

    # Private helper methods
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load metadata file: {e}")
        
        return {
            'checkpoints': [],
            'experiment_id': self.experiment_id,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        try:
            # Create backup of existing metadata
            if self.metadata_file.exists():
                backup_path = self.metadata_file.with_suffix('.json.bak')
                shutil.copy2(self.metadata_file, backup_path)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def _update_metadata(
        self, 
        filename: str, 
        epoch: int, 
        metrics: Dict[str, Any],
        is_best: bool
    ) -> None:
        """Update metadata with new checkpoint information."""
        checkpoint_info = {
            'filename': filename,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'is_best': is_best,
            'file_size': (self.checkpoint_dir / filename).stat().st_size
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        
        # Update best checkpoint tracking
        if is_best:
            self.metadata['best_checkpoint'] = filename
        
        self._save_metadata()

    def _get_main_metric(self, metrics: Dict[str, Any]) -> Optional[float]:
        """Extract main metric for filename generation."""
        # Priority order for common metrics
        priority_metrics = ['loss', 'val_loss', 'reward', 'accuracy', 'mse']
        
        for metric_name in priority_metrics:
            if metric_name in metrics:
                try:
                    return float(metrics[metric_name])
                except (ValueError, TypeError):
                    continue
        
        # Fallback to first numeric metric
        for value in metrics.values():
            try:
                return float(value)
            except (ValueError, TypeError):
                continue
        
        return None

    def _verify_checkpoint(self, checkpoint_path: Path) -> None:
        """Verify that a saved checkpoint can be loaded."""
        try:
            torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise CheckpointCorruptionError(f"Checkpoint verification failed: {e}")

    def _validate_checkpoint_data(self, checkpoint_data: Dict[str, Any]) -> None:
        """Validate the structure of loaded checkpoint data."""
        required_keys = ['model_state_dict', 'epoch']
        
        for key in required_keys:
            if key not in checkpoint_data:
                raise CheckpointCorruptionError(f"Missing required key in checkpoint: {key}")
        
        if not isinstance(checkpoint_data['model_state_dict'], dict):
            raise CheckpointCorruptionError("Invalid model state dict format")

    def __repr__(self) -> str:
        """String representation of CheckpointManager."""
        num_checkpoints = len(self.metadata.get('checkpoints', []))
        return (f"CheckpointManager(dir='{self.checkpoint_dir}', "
                f"experiment_id='{self.experiment_id}', "
                f"checkpoints={num_checkpoints})")


# Utility functions for integration with existing codebase

def create_checkpoint_manager(
    config: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None
) -> CheckpointManager:
    """
    Factory function to create CheckpointManager from configuration.
    
    Args:
        config: Configuration dictionary (should contain 'checkpoint_dir')
        experiment_name: Optional experiment identifier
        
    Returns:
        Configured CheckpointManager instance
    """
    if config is None:
        config = {}
    
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    max_checkpoints = config.get('max_checkpoints', 5)
    auto_cleanup = config.get('auto_cleanup', True)
    
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        experiment_id=experiment_name,
        max_checkpoints=max_checkpoints,
        auto_cleanup=auto_cleanup
    )


def safe_checkpoint_load(
    checkpoint_manager: CheckpointManager,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    fallback_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Safely load checkpoint with fallback options.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        model: Model to load into
        optimizer: Optional optimizer to load into
        fallback_path: Fallback checkpoint path if latest fails
        
    Returns:
        Loaded metadata or None if all loading attempts fail
    """
    try:
        # Try to load latest checkpoint
        return checkpoint_manager.load_checkpoint(model, optimizer, latest=True)
    except CheckpointNotFoundError:
        if fallback_path:
            try:
                return checkpoint_manager.load_checkpoint(
                    model, optimizer, checkpoint_path=fallback_path
                )
            except Exception as e:
                logging.warning(f"Fallback checkpoint loading failed: {e}")
        return None
    except Exception as e:
        logging.warning(f"Latest checkpoint loading failed: {e}")
        return None
