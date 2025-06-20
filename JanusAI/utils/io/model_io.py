"""
Model I/O Utilities
===================

Provides functions for saving and loading machine learning models (e.g., PyTorch models).
This is crucial for checkpointing, resuming training, and deploying trained models.
"""

import torch
import os
from typing import Any, Dict, Optional, Union


class ModelIO:
    """
    A utility class for saving and loading PyTorch models and their states.
    """

    def __init__(self):
        pass # No specific initialization needed for static methods.

    def save_model_state(self, 
                         model: torch.nn.Module, 
                         file_path: str, 
                         optimizer: Optional[torch.optim.Optimizer] = None, 
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Saves the state dictionary of a PyTorch model, and optionally its optimizer.

        Args:
            model: The PyTorch model to save.
            file_path: The path to the .pt or .pth file where the state will be saved.
            optimizer: Optional PyTorch optimizer whose state_dict should also be saved.
            metadata: Optional dictionary of additional metadata (e.g., epoch, loss) to save.
        """
        save_dict = {
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if metadata is not None:
            save_dict['metadata'] = metadata
        
        try:
            torch.save(save_dict, file_path)
            # print(f"Model state saved to {file_path}")
        except Exception as e:
            print(f"Error saving model state to {file_path}: {e}")

    def load_model_state(self, 
                         model: torch.nn.Module, 
                         file_path: str, 
                         optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Loads the state dictionary into a PyTorch model and optionally its optimizer.

        Args:
            model: The PyTorch model instance to load the state into.
            file_path: The path to the .pt or .pth file to load from.
            optimizer: Optional PyTorch optimizer to load state into.

        Returns:
            A dictionary containing loaded metadata (if any), or an empty dict if none.
        """
        if not os.path.exists(file_path):
            print(f"Error: Model state file not found at {file_path}")
            return {}
        
        try:
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # print(f"Model state loaded from {file_path}")
            return checkpoint.get('metadata', {})
        except Exception as e:
            print(f"Error loading model state from {file_path}: {e}")
            return {}

    def save_full_model(self, model: torch.nn.Module, file_path: str):
        """
        Saves the entire PyTorch model (architecture + state).
        NOTE: This approach is less flexible than saving state_dict for deployment
              but can be convenient for saving and loading during development if
              the class definition is always available.
        """
        try:
            torch.save(model, file_path)
            # print(f"Full model saved to {file_path}")
        except Exception as e:
            print(f"Error saving full model to {file_path}: {e}")

    def load_full_model(self, file_path: str) -> Optional[torch.nn.Module]:
        """
        Loads a full PyTorch model (architecture + state).
        Requires the model class definition to be available in the environment.
        """
        if not os.path.exists(file_path):
            print(f"Error: Full model file not found at {file_path}")
            return None
        try:
            return torch.load(file_path)
        except Exception as e:
            print(f"Error loading full model from {file_path}: {e}")
            return None


if __name__ == "__main__":
    print("--- Testing ModelIO Utilities ---")
    model_io = ModelIO()

    # Create a temporary directory for testing
    test_dir = "temp_model_io_test"
    os.makedirs(test_dir, exist_ok=True)

    # Define a simple mock PyTorch model and optimizer
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
            self.relu = torch.nn.ReLU()
        def forward(self, x):
            return self.relu(self.linear(x))

    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- Test Saving and Loading Model State ---
    model_state_file = os.path.join(test_dir, "model_state.pt")
    metadata = {'epoch': 10, 'loss': 0.123}

    model_io.save_model_state(model, model_state_file, optimizer, metadata)
    
    # Create new instances for loading
    new_model = SimpleNet()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001) # Different LR to check if loaded

    loaded_metadata = model_io.load_model_state(new_model, model_state_file, new_optimizer)
    
    print("\nModel State I/O Test:")
    print("Original model param (first 5):", list(model.parameters())[0].data.flatten()[:5])
    print("Loaded model param (first 5):", list(new_model.parameters())[0].data.flatten()[:5])
    assert torch.allclose(list(model.parameters())[0].data, list(new_model.parameters())[0].data)
    
    print("Original optimizer state (first param's momentum):", optimizer.state_dict()['state'][list(optimizer.state_dict()['state'].keys())[0]]['exp_avg'][:5])
    print("Loaded optimizer state (first param's momentum):", new_optimizer.state_dict()['state'][list(new_optimizer.state_dict()['state'].keys())[0]]['exp_avg'][:5])
    assert torch.allclose(optimizer.state_dict()['state'][list(optimizer.state_dict()['state'].keys())[0]]['exp_avg'], new_optimizer.state_dict()['state'][list(new_optimizer.state_dict()['state'].keys())[0]]['exp_avg'])

    print("Loaded Metadata:", loaded_metadata)
    assert loaded_metadata == metadata
    print("Model State I/O: SUCCESS")

    # --- Test Saving and Loading Full Model ---
    full_model_file = os.path.join(test_dir, "full_model.pth")
    model_io.save_full_model(model, full_model_file)
    
    loaded_full_model = model_io.load_full_model(full_model_file)
    print("\nFull Model I/O Test:")
    assert loaded_full_model is not None and isinstance(loaded_full_model, SimpleNet)
    # Check if loaded model weights match original
    assert torch.allclose(list(model.parameters())[0].data, list(loaded_full_model.parameters())[0].data)
    print("Full Model I/O: SUCCESS")

    # --- Test Error Handling (Non-existent files) ---
    print("\nError Handling Test (expect warnings/errors):")
    non_existent_file = os.path.join(test_dir, "non_existent_model.pt")
    loaded_meta_none = model_io.load_model_state(new_model, non_existent_file)
    assert loaded_meta_none == {}
    print("Loading non-existent model state: Handled (returns empty dict)")
    
    loaded_model_none = model_io.load_full_model(non_existent_file)
    assert loaded_model_none is None
    print("Loading non-existent full model: Handled (returns None)")

    # Clean up the temporary directory
    import shutil
    shutil.rmtree(test_dir)
    print(f"\nCleaned up test directory: {test_dir}")

    print("\nAll ModelIO tests completed.")

