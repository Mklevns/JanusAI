
# janus/utils/config/test_validation.py
"""
Comprehensive tests for the @validate_inputs decorator.
Tests type checking, error handling, and edge cases for function input validation.
"""
import pytest
from typing import List, Dict, Optional, Union, Any
import numpy as np

from janus_ai.utils.config.validation import validate_inputs


class TestValidateInputsDecorator:
    """Test the @validate_inputs decorator functionality."""
    
    def test_basic_type_validation(self):
        """Test basic type validation for simple types."""
        @validate_inputs
        def add_numbers(a: int, b: int) -> int:
            return a + b
        
        # Valid inputs
        assert add_numbers(2, 3) == 5
        assert add_numbers(a=10, b=20) == 30
        
        # Invalid inputs
        with pytest.raises(TypeError, match="Argument 'a' expected type int"):
            add_numbers("2", 3)
        
        with pytest.raises(TypeError, match="Argument 'b' expected type int"):
            add_numbers(2, "3")
    
    def test_optional_and_default_parameters(self):
        """Test validation with optional and default parameters."""
        @validate_inputs
        def greet(name: str, title: Optional[str] = None, punctuation: str = "!") -> str:
            if title:
                return f"{title} {name}{punctuation}"
            return f"Hello {name}{punctuation}"
        
        # Valid calls
        assert greet("Alice") == "Hello Alice!"
        assert greet("Bob", "Dr.") == "Dr. Bob!"
        assert greet("Charlie", punctuation="?") == "Hello Charlie?"
        
        # Invalid type for optional parameter
        with pytest.raises(TypeError, match="Argument 'title' expected type"):
            greet("Dave", title=123)
        
        # Invalid type for default parameter
        with pytest.raises(TypeError, match="Argument 'punctuation' expected type str"):
            greet("Eve", punctuation=None)
    
    def test_collection_types(self):
        """Test validation with collection types."""
        @validate_inputs
        def process_data(
            numbers: List[int],
            config: Dict[str, float],
            tags: Optional[List[str]] = None
        ) -> float:
            result = sum(numbers) * sum(config.values())
            if tags:
                result *= len(tags)
            return result
        
        # Valid inputs
        assert process_data([1, 2, 3], {"scale": 0.5, "offset": 1.0}) == 9.0
        assert process_data([1, 2], {"a": 1.0}, ["tag1", "tag2"]) == 6.0
        
        # Invalid list type
        with pytest.raises(TypeError, match="Argument 'numbers' expected type"):
            process_data("not a list", {"a": 1.0})
        
        # Invalid dict type
        with pytest.raises(TypeError, match="Argument 'config' expected type"):
            process_data([1, 2], "not a dict")
        
        # Note: The decorator may not validate inner types of collections
        # This is a limitation of simple type checking
    
    def test_union_types(self):
        """Test validation with Union types."""
        @validate_inputs
        def flexible_add(a: Union[int, float], b: Union[int, float]) -> float:
            return float(a + b)
        
        # Valid inputs
        assert flexible_add(1, 2) == 3.0
        assert flexible_add(1.5, 2.5) == 4.0
        assert flexible_add(1, 2.5) == 3.5
        
        # Invalid inputs
        with pytest.raises(TypeError, match="Argument 'a' expected type"):
            flexible_add("1", 2)
    
    def test_args_and_kwargs(self):
        """Test validation with *args and **kwargs."""
        @validate_inputs
        def variadic_function(
            base: int,
            *numbers: int,
            multiplier: float = 1.0,
            **options: str
        ) -> float:
            total = base + sum(numbers)
            return total * multiplier
        
        # Valid calls
        assert variadic_function(10) == 10.0
        assert variadic_function(10, 1, 2, 3) == 16.0
        assert variadic_function(10, 5, multiplier=2.0) == 30.0
        assert variadic_function(10, 1, 2, multiplier=0.5, debug="true") == 6.5
        
        # Invalid base type
        with pytest.raises(TypeError, match="Argument 'base' expected type int"):
            variadic_function("10")
        
        # Invalid multiplier type
        with pytest.raises(TypeError, match="Argument 'multiplier' expected type float"):
            variadic_function(10, multiplier="2.0")
    
    def test_class_methods(self):
        """Test decorator on class methods."""
        class Calculator:
            @validate_inputs
            def __init__(self, precision: int = 2):
                self.precision = precision
            
            @validate_inputs
            def add(self, a: float, b: float) -> float:
                return round(a + b, self.precision)
            
            @validate_inputs
            @classmethod
            def from_string(cls, value: str) -> 'Calculator':
                return cls(precision=int(value))
            
            @validate_inputs
            @staticmethod
            def validate_number(n: Union[int, float]) -> bool:
                return isinstance(n, (int, float))
        
        # Test instance method
        calc = Calculator(precision=3)
        assert calc.add(1.1111, 2.2222) == 3.333
        
        with pytest.raises(TypeError):
            calc.add("1", 2)
        
        # Test class method
        calc2 = Calculator.from_string("4")
        assert calc2.precision == 4
        
        # Test static method
        assert Calculator.validate_number(42) is True
        assert Calculator.validate_number(3.14) is True
    
    def test_numpy_types(self):
        """Test validation with numpy types."""
        @validate_inputs
        def process_array(
            data: np.ndarray,
            scalar: Union[int, float, np.number]
        ) -> np.ndarray:
            return data * scalar
        
        # Valid inputs
        arr = np.array([1, 2, 3])
        assert np.array_equal(process_array(arr, 2), np.array([2, 4, 6]))
        assert np.array_equal(process_array(arr, 2.5), np.array([2.5, 5.0, 7.5]))
        assert np.array_equal(process_array(arr, np.int64(2)), np.array([2, 4, 6]))
        
        # Invalid array type
        with pytest.raises(TypeError):
            process_array([1, 2, 3], 2)  # List instead of ndarray
    
    def test_any_type(self):
        """Test that Any type allows anything."""
        @validate_inputs
        def accept_anything(value: Any) -> str:
            return f"Got: {type(value).__name__}"
        
        # All of these should work
        assert accept_anything(42) == "Got: int"
        assert accept_anything("hello") == "Got: str"
        assert accept_anything([1, 2, 3]) == "Got: list"
        assert accept_anything(None) == "Got: NoneType"
    
    def test_missing_annotations(self):
        """Test behavior with missing type annotations."""
        @validate_inputs
        def partially_annotated(a: int, b, c: str = "default"):
            return f"{a}-{b}-{c}"
        
        # Should validate annotated parameters only
        assert partially_annotated(1, "anything", "test") == "1-anything-test"
        assert partially_annotated(1, 42) == "1-42-default"
        assert partially_annotated(1, [1, 2, 3]) == "1-[1, 2, 3]-default"
        
        # Should still validate the annotated parameter
        with pytest.raises(TypeError, match="Argument 'a' expected type int"):
            partially_annotated("not an int", "b value")
    
    def test_recursive_validation(self):
        """Test that decorator doesn't interfere with recursion."""
        @validate_inputs
        def factorial(n: int) -> int:
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        
        assert factorial(5) == 120
        assert factorial(0) == 1
        
        with pytest.raises(TypeError):
            factorial("5")
    
    def test_generator_functions(self):
        """Test decorator on generator functions."""
        @validate_inputs
        def count_up_to(n: int):
            for i in range(n):
                yield i
        
        # Valid usage
        assert list(count_up_to(5)) == [0, 1, 2, 3, 4]
        
        # Invalid usage
        with pytest.raises(TypeError):
            list(count_up_to("5"))
    
    def test_complex_nested_types(self):
        """Test with complex nested type hints."""
        @validate_inputs
        def complex_function(
            data: Dict[str, List[Union[int, float]]],
            optional_config: Optional[Dict[str, Any]] = None
        ) -> List[float]:
            result = []
            for key, values in data.items():
                result.extend(float(v) for v in values)
            return result
        
        # Valid inputs
        test_data = {"a": [1, 2.5], "b": [3, 4.0]}
        assert complex_function(test_data) == [1.0, 2.5, 3.0, 4.0]
        assert complex_function(test_data, {"debug": True}) == [1.0, 2.5, 3.0, 4.0]
        
        # Invalid inputs
        with pytest.raises(TypeError):
            complex_function("not a dict")
    
    def test_error_messages(self):
        """Test that error messages are informative."""
        @validate_inputs
        def typed_function(name: str, age: int, score: float) -> str:
            return f"{name} ({age}) scored {score}"
        
        # Test each parameter
        with pytest.raises(TypeError) as exc_info:
            typed_function(123, 25, 95.5)
        assert "Argument 'name' expected type str, but got int" in str(exc_info.value)
        
        with pytest.raises(TypeError) as exc_info:
            typed_function("Alice", "25", 95.5)
        assert "Argument 'age' expected type int, but got str" in str(exc_info.value)
        
        with pytest.raises(TypeError) as exc_info:
            typed_function("Alice", 25, "95.5")
        assert "Argument 'score' expected type float, but got str" in str(exc_info.value)
    
    def test_preserve_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @validate_inputs
        def documented_function(x: int) -> int:
            """This function doubles the input."""
            return x * 2
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function doubles the input."
        
        # Should still have annotations
        assert hasattr(documented_function, '__annotations__')