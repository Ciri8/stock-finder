"""
Main test module that imports and runs all model tests.
This file can be used as an entry point for pytest or unittest discovery.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test classes
from test_pattern_recognition import TestPatternRecognition, TestPatternRecognitionIntegration
from test_data_splitter import TestDataSplitter, TestDataSplitterIntegration
from test_integration import TestPatternRecognitionPipeline, TestRealWorldIntegration


def suite():
    """Create and return a test suite with all tests."""
    test_suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    # Add all test classes
    test_suite.addTests(loader.loadTestsFromTestCase(TestPatternRecognition))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPatternRecognitionIntegration))
    test_suite.addTests(loader.loadTestsFromTestCase(TestDataSplitter))
    test_suite.addTests(loader.loadTestsFromTestCase(TestDataSplitterIntegration))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPatternRecognitionPipeline))
    test_suite.addTests(loader.loadTestsFromTestCase(TestRealWorldIntegration))
    
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())