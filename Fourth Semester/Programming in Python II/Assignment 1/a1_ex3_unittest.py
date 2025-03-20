import unittest
import os
import shutil
import time
import io
import sys
from unittest.mock import patch
from a1_ex3 import expensive_calculation

class TestExpensiveCalculationCache(unittest.TestCase):
    cache_dir = "cache_txt"
    
    def setUp(self):
        # Remove cache directory if it exists.
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
    
    def tearDown(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_calculation_correct(self, _):
        result = expensive_calculation(2, 3)
        self.assertEqual(result, "The result is 6")
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_cache_file_created(self, _):
        expensive_calculation(10, 20)
        expected_filename = os.path.join(self.cache_dir, "expensive_calculation_10_20_.txt")
        self.assertTrue(os.path.exists(expected_filename))
    
    def test_cached_call_fast(self):
        start = time.time()
        result1 = expensive_calculation(4, 5)
        duration1 = time.time() - start
        
        start = time.time()
        result2 = expensive_calculation(4, 5)
        duration2 = time.time() - start
        
        self.assertEqual(result1, result2)
        # The second call should be much faster.
        self.assertTrue(duration2 < duration1)
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_different_arguments_produce_different_cache(self, _):
        expensive_calculation(3, 3)
        expensive_calculation(3, 4)
        filename1 = os.path.join(self.cache_dir, "expensive_calculation_3_3_.txt")
        filename2 = os.path.join(self.cache_dir, "expensive_calculation_3_4_.txt")
        self.assertNotEqual(filename1, filename2)
        self.assertTrue(os.path.exists(filename1))
        self.assertTrue(os.path.exists(filename2))
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_output_consistency(self, _):
        result1 = expensive_calculation(7, 8)
        result2 = expensive_calculation(7, 8)
        self.assertEqual(result1, result2)
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_cache_directory_created(self, _):
        self.assertFalse(os.path.exists(self.cache_dir))
        expensive_calculation(5, 5)
        self.assertTrue(os.path.exists(self.cache_dir))
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_cache_content(self, _):
        expected_output = "The result is 12"
        expensive_calculation(3, 4)
        filename = os.path.join(self.cache_dir, "expensive_calculation_3_4_.txt")
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, expected_output)
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_multiple_calls_with_different_args(self, _):
        args_list = [(2,2), (3,3), (4,4)]
        results = [expensive_calculation(x, y) for x, y in args_list]
        expected = [f"The result is {x*y}" for x, y in args_list]
        self.assertEqual(results, expected)
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_print_cache_message(self, _):
        # Capture stdout to check for the cache message on the second call.
        captured_output = io.StringIO()
        sys.stdout = captured_output
        expensive_calculation(6, 7)  # First call (no cache message).
        captured_output.truncate(0)
        captured_output.seek(0)
        expensive_calculation(6, 7)  # Second call should print a cache message.
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("[CACHE] Using cached result", output)
    
    @patch('a1_ex3.time.sleep', return_value=None)
    def test_function_output_after_cache(self, _):
        # Call the function twice with new arguments.
        result1 = expensive_calculation(5, 15)
        result2 = expensive_calculation(5, 15)
        self.assertEqual(result1, "The result is 75")
        self.assertEqual(result2, "The result is 75")

if __name__ == '__main__':
    unittest.main()
