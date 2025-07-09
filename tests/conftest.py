"""Test configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest

from codechat.config import RANDOM_SEED


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_python_code():
    """Provide sample Python code for testing."""
    return '''def fibonacci(n):
    """Calculate the nth Fibonacci number.
    
    Args:
        n: The position in the Fibonacci sequence.
        
    Returns:
        The nth Fibonacci number.
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, value):
        """Add a value to the result."""
        self.result += value
        return self.result
    
    def multiply(self, value):
        """Multiply the result by a value."""
        self.result *= value
        return self.result
    
    def reset(self):
        """Reset the calculator."""
        self.result = 0
'''


@pytest.fixture
def sample_javascript_code():
    """Provide sample JavaScript code for testing."""
    return '''function fibonacci(n) {
    /**
     * Calculate the nth Fibonacci number.
     * @param {number} n - The position in the Fibonacci sequence.
     * @returns {number} The nth Fibonacci number.
     */
    if (n <= 0) {
        return 0;
    } else if (n === 1) {
        return 1;
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

class Calculator {
    /**
     * A simple calculator class.
     */
    constructor() {
        this.result = 0;
    }
    
    add(value) {
        /**
         * Add a value to the result.
         * @param {number} value - The value to add.
         * @returns {number} The new result.
         */
        this.result += value;
        return this.result;
    }
    
    multiply(value) {
        /**
         * Multiply the result by a value.
         * @param {number} value - The value to multiply by.
         * @returns {number} The new result.
         */
        this.result *= value;
        return this.result;
    }
    
    reset() {
        /**
         * Reset the calculator.
         */
        this.result = 0;
    }
}
'''
