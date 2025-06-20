"""
Custom Exception Classes for Janus Framework

This module defines custom exception classes specific to the Janus framework to enable
more precise and meaningful error handling throughout the codebase. All custom exceptions
inherit from JanusError to allow for unified exception handling.

The exception hierarchy is designed to provide:
- Improved readability and semantic meaning
- Granular error handling capabilities
- Better debugging and error reporting
- Clear API contracts for different failure modes

Exception Hierarchy:
    JanusError (base)
    ├── ConfigurationError
    │   ├── InvalidConfigError
    │   ├── MissingConfigError
    │   └── ConfigValidationError
    ├── ExperimentError
    │   ├── ExperimentNotFoundError
    │   ├── InvalidExperimentError
    │   └── ExperimentStateError
    ├── GrammarError
    │   ├── ExpressionParsingError
    │   ├── GrammarRuleError
    │   └── SymbolicMathError
    ├── DataError
    │   ├── DataGenerationError
    │   ├── DataValidationError
    │   └── DataCorruptionError
    ├── ModelError
    │   ├── ModelLoadingError
    │   ├── ModelArchitectureError
    │   └── CheckpointError
    ├── PluginError
    │   ├── PluginNotFoundError
    │   ├── MissingDependencyError
    │   └── PluginLoadingError
    ├── EnvironmentError (for RL environments)
    │   ├── EnvironmentSetupError
    │   └── UnsupportedOperationError
    └── TrainingError
        ├── TrainingInterruptedError
        ├── OptimizationError
        └── MetricComputationError
"""

from typing import Optional, Any, Dict, List, Union
import traceback


class JanusError(Exception):
    """
    Base exception for all Janus-specific errors.
    
    This is the root of the custom exception hierarchy. All other Janus exceptions
    should inherit from this class to enable unified exception handling.
    
    Attributes:
        message: Primary error message
        context: Additional context information (optional)
        suggestion: Suggested action to resolve the error (optional)
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize JanusError.
        
        Args:
            message: Primary error message describing what went wrong
            context: Additional context information for debugging
            suggestion: Suggested action to resolve the error
            cause: Original exception that caused this error (if any)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        self.cause = cause
        
        # Store the original traceback if there was a cause
        if cause:
            self.original_traceback = traceback.format_exc()
        else:
            self.original_traceback = None
    
    def __str__(self) -> str:
        """Return a comprehensive string representation of the error."""
        result = self.message
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            result += f" (Context: {context_str})"
        
        if self.suggestion:
            result += f" | Suggestion: {self.suggestion}"
            
        return result
    
    def get_detailed_message(self) -> str:
        """Return a detailed error message including context and suggestions."""
        lines = [f"{self.__class__.__name__}: {self.message}"]
        
        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")
        
        if self.suggestion:
            lines.append(f"Suggestion: {self.suggestion}")
        
        if self.cause:
            lines.append(f"Caused by: {type(self.cause).__name__}: {self.cause}")
        
        return "\n".join(lines)


# Configuration-related exceptions
class ConfigurationError(JanusError):
    """Base class for configuration-related errors."""
    pass


class InvalidConfigError(ConfigurationError):
    """
    Raised when configuration data is invalid or malformed.
    
    This exception is used when configuration parameters have invalid values,
    types, or combinations that prevent proper operation.
    """
    
    def __init__(
        self, 
        message: str,
        config_field: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        expected_type: Optional[type] = None,
        **kwargs
    ):
        context = {}
        if config_field:
            context['field'] = config_field
        if invalid_value is not None:
            context['invalid_value'] = invalid_value
        if expected_type:
            context['expected_type'] = expected_type.__name__
            
        suggestion = "Check configuration file format and parameter values"
        if config_field:
            suggestion += f" for field '{config_field}'"
            
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, missing_field: str, config_file: Optional[str] = None, **kwargs):
        message = f"Missing required configuration field: '{missing_field}'"
        context = {'missing_field': missing_field}
        if config_file:
            context['config_file'] = config_file
            
        suggestion = f"Add '{missing_field}' to your configuration file"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class ConfigValidationError(ConfigurationError):
    """Raised when configuration fails validation rules."""
    
    def __init__(self, validation_errors: List[str], **kwargs):
        message = f"Configuration validation failed: {len(validation_errors)} error(s)"
        context = {'validation_errors': validation_errors}
        suggestion = "Fix the validation errors listed in the context"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


# Experiment-related exceptions
class ExperimentError(JanusError):
    """Base class for experiment-related errors."""
    pass


class ExperimentNotFoundError(ExperimentError):
    """Raised when a requested experiment cannot be found."""
    
    def __init__(self, experiment_id: str, search_location: Optional[str] = None, **kwargs):
        message = f"Experiment not found: '{experiment_id}'"
        context = {'experiment_id': experiment_id}
        if search_location:
            context['search_location'] = search_location
            
        suggestion = "Check experiment ID and ensure experiment has been created"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class InvalidExperimentError(ExperimentError):
    """Raised when experiment definition or state is invalid."""
    
    def __init__(self, experiment_id: str, reason: str, **kwargs):
        message = f"Invalid experiment '{experiment_id}': {reason}"
        context = {'experiment_id': experiment_id, 'reason': reason}
        suggestion = "Review experiment configuration and state"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class ExperimentStateError(ExperimentError):
    """Raised when experiment is in an invalid state for the requested operation."""
    
    def __init__(
        self, 
        experiment_id: str, 
        current_state: str, 
        required_state: str,
        operation: str,
        **kwargs
    ):
        message = f"Cannot perform '{operation}' on experiment '{experiment_id}' in state '{current_state}'"
        context = {
            'experiment_id': experiment_id,
            'current_state': current_state,
            'required_state': required_state,
            'operation': operation
        }
        suggestion = f"Ensure experiment is in state '{required_state}' before performing '{operation}'"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


# Grammar and symbolic math exceptions
class GrammarError(JanusError):
    """Base class for grammar and symbolic math related errors."""
    pass


class ExpressionParsingError(GrammarError):
    """Raised when an expression cannot be parsed according to grammar rules."""
    
    def __init__(self, expression: str, grammar_type: Optional[str] = None, **kwargs):
        message = f"Failed to parse expression: '{expression}'"
        context = {'expression': expression}
        if grammar_type:
            context['grammar_type'] = grammar_type
            
        suggestion = "Check expression syntax and ensure it conforms to the grammar rules"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class GrammarRuleError(GrammarError):
    """Raised when there's an issue with grammar rule definition or application."""
    
    def __init__(self, rule_name: str, issue: str, **kwargs):
        message = f"Grammar rule error in '{rule_name}': {issue}"
        context = {'rule_name': rule_name, 'issue': issue}
        suggestion = "Review grammar rule definition and ensure it's properly formatted"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class SymbolicMathError(GrammarError):
    """Raised when symbolic mathematics operations fail."""
    
    def __init__(self, operation: str, details: str, **kwargs):
        message = f"Symbolic math operation failed: {operation}"
        context = {'operation': operation, 'details': details}
        suggestion = "Check mathematical expression validity and variable definitions"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


# Data-related exceptions
class DataError(JanusError):
    """Base class for data-related errors."""
    pass


class DataGenerationError(DataError):
    """Raised when there's a problem generating data for physics tasks or environments."""
    
    def __init__(
        self, 
        task_name: Optional[str] = None, 
        generation_step: Optional[str] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        message = "Data generation failed"
        if task_name:
            message += f" for task '{task_name}'"
        if generation_step:
            message += f" during step '{generation_step}'"
        if details:
            message += f": {details}"
            
        context = {}
        if task_name:
            context['task_name'] = task_name
        if generation_step:
            context['generation_step'] = generation_step
        if details:
            context['details'] = details
            
        suggestion = "Check data generation parameters and numerical stability"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class DataValidationError(DataError):
    """Raised when data fails validation checks."""
    
    def __init__(self, validation_issue: str, data_source: Optional[str] = None, **kwargs):
        message = f"Data validation failed: {validation_issue}"
        context = {'validation_issue': validation_issue}
        if data_source:
            context['data_source'] = data_source
            
        suggestion = "Review data quality and validation criteria"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class DataCorruptionError(DataError):
    """Raised when data appears to be corrupted or malformed."""
    
    def __init__(self, data_type: str, corruption_details: str, **kwargs):
        message = f"Data corruption detected in {data_type}: {corruption_details}"
        context = {'data_type': data_type, 'corruption_details': corruption_details}
        suggestion = "Regenerate or reload the corrupted data"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


# Model-related exceptions
class ModelError(JanusError):
    """Base class for model-related errors."""
    pass


class ModelLoadingError(ModelError):
    """Raised when there's an issue loading a pre-trained model or checkpoint."""
    
    def __init__(
        self, 
        model_path: str, 
        error_type: str = "loading",
        details: Optional[str] = None,
        **kwargs
    ):
        message = f"Model {error_type} failed for '{model_path}'"
        if details:
            message += f": {details}"
            
        context = {'model_path': model_path, 'error_type': error_type}
        if details:
            context['details'] = details
            
        suggestion = "Check model file existence, format, and compatibility"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class ModelArchitectureError(ModelError):
    """Raised when there's an issue with model architecture definition or compatibility."""
    
    def __init__(self, architecture_issue: str, model_name: Optional[str] = None, **kwargs):
        message = f"Model architecture error: {architecture_issue}"
        context = {'architecture_issue': architecture_issue}
        if model_name:
            context['model_name'] = model_name
            
        suggestion = "Review model architecture definition and parameter compatibility"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class CheckpointError(ModelError):
    """Raised when there's an issue with checkpoint operations."""
    
    def __init__(
        self, 
        operation: str, 
        checkpoint_path: Optional[str] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        message = f"Checkpoint {operation} failed"
        if checkpoint_path:
            message += f" for '{checkpoint_path}'"
        if details:
            message += f": {details}"
            
        context = {'operation': operation}
        if checkpoint_path:
            context['checkpoint_path'] = checkpoint_path
        if details:
            context['details'] = details
            
        suggestion = "Check file permissions, disk space, and checkpoint format"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


# Plugin-related exceptions
class PluginError(JanusError):
    """Base class for plugin-related errors."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a specified plugin cannot be found."""
    
    def __init__(
        self, 
        plugin_name: str, 
        plugin_type: Optional[str] = None,
        available_plugins: Optional[List[str]] = None,
        **kwargs
    ):
        message = f"Plugin not found: '{plugin_name}'"
        if plugin_type:
            message += f" (type: {plugin_type})"
            
        context = {'plugin_name': plugin_name}
        if plugin_type:
            context['plugin_type'] = plugin_type
        if available_plugins:
            context['available_plugins'] = available_plugins
            
        suggestion = "Check plugin name spelling and ensure plugin is installed"
        if available_plugins:
            suggestion += f". Available plugins: {', '.join(available_plugins)}"
            
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class MissingDependencyError(PluginError):
    """Raised when a required dependency is missing."""
    
    def __init__(
        self, 
        dependency_name: str, 
        required_for: Optional[str] = None,
        install_command: Optional[str] = None,
        **kwargs
    ):
        message = f"Missing required dependency: '{dependency_name}'"
        if required_for:
            message += f" (required for {required_for})"
            
        context = {'dependency_name': dependency_name}
        if required_for:
            context['required_for'] = required_for
        if install_command:
            context['install_command'] = install_command
            
        suggestion = f"Install missing dependency: {dependency_name}"
        if install_command:
            suggestion += f" using: {install_command}"
            
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class PluginLoadingError(PluginError):
    """Raised when a plugin fails to load properly."""
    
    def __init__(self, plugin_name: str, loading_error: str, **kwargs):
        message = f"Failed to load plugin '{plugin_name}': {loading_error}"
        context = {'plugin_name': plugin_name, 'loading_error': loading_error}
        suggestion = "Check plugin code for errors and ensure all dependencies are available"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


# Environment-related exceptions (for RL environments)
class EnvironmentError(JanusError):
    """Base class for RL environment-related errors."""
    pass


class EnvironmentSetupError(EnvironmentError):
    """Raised when an environment fails to set up properly."""
    
    def __init__(self, env_name: str, setup_issue: str, **kwargs):
        message = f"Environment setup failed for '{env_name}': {setup_issue}"
        context = {'env_name': env_name, 'setup_issue': setup_issue}
        suggestion = "Check environment configuration and dependencies"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class UnsupportedOperationError(EnvironmentError):
    """Raised when a requested operation is not supported."""
    
    def __init__(
        self, 
        operation: str, 
        context_info: Optional[str] = None,
        alternative: Optional[str] = None,
        **kwargs
    ):
        message = f"Unsupported operation: '{operation}'"
        if context_info:
            message += f" in context: {context_info}"
            
        context = {'operation': operation}
        if context_info:
            context['context_info'] = context_info
        if alternative:
            context['alternative'] = alternative
            
        suggestion = "Use a supported operation or feature"
        if alternative:
            suggestion += f". Consider using: {alternative}"
            
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


# Training-related exceptions
class TrainingError(JanusError):
    """Base class for training-related errors."""
    pass


class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted unexpectedly."""
    
    def __init__(
        self, 
        interruption_reason: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        **kwargs
    ):
        message = f"Training interrupted: {interruption_reason}"
        context = {'interruption_reason': interruption_reason}
        if epoch is not None:
            context['epoch'] = epoch
        if step is not None:
            context['step'] = step
            
        suggestion = "Review logs and consider resuming from last checkpoint"
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class OptimizationError(TrainingError):
    """Raised when optimization process fails."""
    
    def __init__(self, optimizer_issue: str, **kwargs):
        message = f"Optimization failed: {optimizer_issue}"
        context = {'optimizer_issue': optimizer_issue}
        suggestion = "Check learning rate, gradients, and optimization parameters"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


class MetricComputationError(TrainingError):
    """Raised when metric computation fails during training."""
    
    def __init__(self, metric_name: str, computation_error: str, **kwargs):
        message = f"Metric computation failed for '{metric_name}': {computation_error}"
        context = {'metric_name': metric_name, 'computation_error': computation_error}
        suggestion = "Check metric implementation and input data validity"
        
        super().__init__(message, context=context, suggestion=suggestion, **kwargs)


# Utility functions for exception handling
def handle_janus_exception(func):
    """
    Decorator to standardize Janus exception handling.
    
    This decorator catches standard Python exceptions and converts them to
    appropriate Janus exceptions with additional context.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except JanusError:
            # Re-raise Janus exceptions as-is
            raise
        except FileNotFoundError as e:
            if 'config' in str(e).lower():
                raise MissingConfigError(str(e), cause=e)
            elif 'model' in str(e).lower() or 'checkpoint' in str(e).lower():
                raise ModelLoadingError(str(e), cause=e)
            else:
                raise JanusError(f"File not found: {e}", cause=e)
        except ValueError as e:
            if 'config' in str(e).lower():
                raise InvalidConfigError(str(e), cause=e)
            else:
                raise DataValidationError(str(e), cause=e)
        except ImportError as e:
            raise MissingDependencyError(str(e), cause=e)
        except Exception as e:
            # Wrap any other exception as a generic JanusError
            raise JanusError(f"Unexpected error in {func.__name__}: {e}", cause=e)
    
    return wrapper


def format_exception_chain(exception: Exception) -> str:
    """
    Format an exception chain for better readability.
    
    Args:
        exception: The exception to format
        
    Returns:
        Formatted string showing the exception chain
    """
    lines = []
    current = exception
    
    while current:
        if isinstance(current, JanusError):
            lines.append(current.get_detailed_message())
        else:
            lines.append(f"{type(current).__name__}: {current}")
        
        # Follow the chain
        current = getattr(current, 'cause', None) or getattr(current, '__cause__', None)
        if current:
            lines.append("  Caused by:")
    
    return "\n".join(lines)


def raise_with_context(
    exception_class: type, 
    message: str, 
    **context_kwargs
) -> None:
    """
    Raise an exception with additional context information.
    
    Args:
        exception_class: The exception class to raise
        message: The error message
        **context_kwargs: Additional context to include in the exception
    """
    if issubclass(exception_class, JanusError):
        raise exception_class(message, context=context_kwargs)
    else:
        raise exception_class(message)


# Legacy compatibility - maintain existing exception names used in codebase
# These are aliases to maintain compatibility with existing code
InvalidConfigError = InvalidConfigError  # Already defined above
DataGenerationError = DataGenerationError  # Already defined above
PluginNotFoundError = PluginNotFoundError  # Already defined above
MissingDependencyError = MissingDependencyError  # Already defined above
