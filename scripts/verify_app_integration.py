import unittest
import os
import sys
import numpy as np
from PIL import Image
import io

# Add current directory to path
sys.path.append(os.getcwd())

from app import app, CLASS_NAMES, USE_EXPANDED_DATASET

class TestAppIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_configuration(self):
        """Test application configuration"""
        print("\n[TEST] Verifying Configuration...")
        self.assertTrue(USE_EXPANDED_DATASET, "USE_EXPANDED_DATASET should be True")
        self.assertEqual(len(CLASS_NAMES), 100, f"Expected 100 classes, found {len(CLASS_NAMES)}")
        print(f"[OK] Configuration verified: 100 classes enabled")

    def test_health_check(self):
        """Test health check endpoint"""
        print("\n[TEST] Verifying Health Check...")
        response = self.app.get('/api/health')
        data = response.get_json()
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'], "Model should be loaded")
        print("[OK] Health check passed: Model is loaded")

    def test_classes_endpoint(self):
        """Test classes endpoint"""
        print("\n[TEST] Verifying Classes Endpoint...")
        response = self.app.get('/api/classes')
        data = response.get_json()
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['count'], 100)
        self.assertEqual(len(data['classes']), 100)
        print("[OK] Classes endpoint verified")

    def test_nutrition_endpoint(self):
        """Test nutrition endpoint for a new dish"""
        print("\n[TEST] Verifying Nutrition Endpoint...")
        dish_name = "Pizza"  # One of the new dishes
        response = self.app.get(f'/api/nutrition/{dish_name}')
        data = response.get_json()
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['dish_name'], dish_name)
        self.assertIn('calories', data['nutrition']['nutrition'])
        print(f"[OK] Nutrition data verified for {dish_name}")

if __name__ == '__main__':
    unittest.main()
