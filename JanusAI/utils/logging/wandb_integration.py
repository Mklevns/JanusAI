"""
Weights & Biases Integration
============================

Provides utilities for integrating training runs with Weights & Biases (W&B)
for experiment tracking, visualization, and collaboration.
"""

# Use safe_import to make wandb an optional dependency
from JanusAI.utils.general_utils import safe_import
wandb = safe_import("wandb", "wandb")

from typing import Any, Dict, Optional, Union, List


class WandbLogger:
    """
    A utility class for logging metrics and artifacts to Weights & Biases.
    Provides a thin wrapper around W&B functionalities.
    """

    def __init__(self, 
                 project: str, 
                 entity: Optional[str] = None, 
                 name: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 mode: str = "online"):
        """
        Initializes the W&B logger.

        Args:
            project: The name of the W&B project.
            entity: The W&B entity (username or team name).
            name: An optional display name for the run.
            config: A dictionary of hyperparameters or experiment configurations to log.
            tags: A list of strings to tag the run.
            mode: W&B run mode ('online', 'offline', 'disabled').
        """
        self.is_enabled = wandb is not None and mode != "disabled"
        if not self.is_enabled:
            print("Warning: Weights & Biases is not enabled (module not found or mode is 'disabled'). Logging will be skipped.")
            return

        try:
            wandb.init(project=project, entity=entity, name=name, config=config, tags=tags, mode=mode)
            self.run = wandb.run
            print(f"Weights & Biases run initialized: {self.run.url}")
        except Exception as e:
            self.is_enabled = False
            print(f"Error initializing Weights & Biases: {e}. W&B logging disabled.")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Logs a dictionary of metrics to W&B.

        Args:
            metrics: A dictionary of metrics to log (e.g., {'loss': 0.1, 'accuracy': 0.9}).
            step: Optional global step/iteration number for logging.
        """
        if self.is_enabled and self.run:
            try:
                self.run.log(metrics, step=step)
            except Exception as e:
                print(f"Error logging metrics to W&B: {e}")
                self.is_enabled = False # Disable further logging if error persists

    def log_config(self, config: Dict[str, Any]):
        """Logs additional configuration parameters."""
        if self.is_enabled and self.run:
            try:
                self.run.config.update(config)
            except Exception as e:
                print(f"Error updating W&B config: {e}")

    def watch_model(self, model: torch.nn.Module, log: str = "gradients", log_freq: int = 100):
        """
        Starts watching a PyTorch model for gradient and parameter logging.

        Args:
            model: The PyTorch model to watch.
            log: What to log ('gradients', 'parameters', 'all').
            log_freq: How often to log (steps).
        """
        if self.is_enabled and self.run:
            try:
                wandb.watch(model, log=log, log_freq=log_freq)
                print("W&B is watching the model.")
            except Exception as e:
                print(f"Error setting up W&B model watch: {e}")

    def finish(self):
        """Finishes the W&B run."""
        if self.is_enabled and self.run:
            try:
                wandb.finish()
                print("Weights & Biases run finished.")
            except Exception as e:
                print(f"Error finishing W&B run: {e}")
            self.is_enabled = False
            self.run = None # Clear run reference


if __name__ == "__main__":
    print("--- Testing WandbLogger (requires wandb to be installed) ---")

    # This test will only fully run if wandb is installed (`pip install wandb`)
    # and you are logged in (`wandb login`).
    # Otherwise, it will print warnings and disable itself.

    project_name = "janus_test_project"
    run_name = "test_run_" + datetime.now().strftime('%H%M%S')
    test_config = {"learning_rate": 0.001, "epochs": 5}

    logger = WandbLogger(
        project=project_name,
        name=run_name,
        config=test_config,
        tags=["dev", "test"],
        mode="online" # Use "offline" if you don't want to sync to cloud immediately
    )

    if logger.is_enabled:
        # Simulate a training loop
        for i in range(10):
            mock_loss = 1.0 / (i + 1) + np.random.rand() * 0.01
            mock_accuracy = 0.5 + i * 0.05 + np.random.rand() * 0.02
            
            logger.log({"loss": mock_loss, "accuracy": mock_accuracy}, step=i)
            print(f"Logged step {i}: loss={mock_loss:.4f}, accuracy={mock_accuracy:.4f}")
            time.sleep(0.1)

        # Log some final metrics
        logger.log({"final_loss": mock_loss, "final_accuracy": mock_accuracy}, step=i + 1)
        
        # Log an artifact (e.g., a dummy file)
        with open("dummy_artifact.txt", "w") as f:
            f.write("This is a test artifact.")
        if wandb: # Check if wandb was imported successfully
            try:
                artifact = wandb.Artifact("my-artifact", type="dataset")
                artifact.add_file("dummy_artifact.txt")
                wandb.log_artifact(artifact)
                print("\nLogged dummy_artifact.txt to W&B.")
            except Exception as e:
                print(f"Error logging artifact: {e}")
        
        logger.finish()
        os.remove("dummy_artifact.txt") # Clean up dummy file
    else:
        print("\nSkipping full W&B test as it's not enabled.")

    print("\nWandbLogger tests completed.")

