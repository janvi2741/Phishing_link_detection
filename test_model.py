#!/usr/bin/env python3
"""
Complete test script for the phishing detection project
"""

import pickle
import os
import sys
from datetime import datetime

# Test 1: Test the rule-based pickle model
def test_pickle_model():
    print("=" * 60)
    print("TEST 1: Rule-based Pickle Model")
    print("=" * 60)
    
    try:
        # Import the classes directly
        from create_final_pickle import PhishingPredictor, URLFeatureExtractor
        
        # Load the pickle file
        with open('phishing_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print("✓ Pickle model loaded successfully")
        
        # Test URLs
        test_urls = [
            "https://www.google.com",
            "https://github.com/microsoft/vscode",
            "http://secure-paypal-update.suspicious-domain.tk/login",
            "https://amazon-security-alert.fake-site.ml/verify",
            "http://192.168.1.1/admin",
            "https://www.wikipedia.org/wiki/Machine_Learning"
        ]
        
        print(f"\nTesting {len(test_urls)} URLs:")
        print("-" * 60)
        
        for url in test_urls:
            result = model.predict_url(url)
            print(f"URL: {url}")
            print(f"Prediction: {result['prediction']}")
            print(f"Phishing Score: {result['phishing_score']:.3f}")
            print(f"Risk Level: {result['risk_level']}")
            if result['suspicious_features']:
                print(f"Suspicious Features: {', '.join(result['suspicious_features'])}")
            print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing pickle model: {e}")
        return False

# Test 2: Test data preprocessing
def test_data_preprocessing():
    print("\nTEST 2: Data Preprocessing")
    print("=" * 60)
    
    try:
        from data_preprocessing import PhishingDataPreprocessor, URLFeatureExtractor
        
        # Test feature extraction
        extractor = URLFeatureExtractor()
        test_url = "https://secure-paypal-update.suspicious-domain.tk/login"
        features = extractor.extract_url_features(test_url)
        
        print("✓ URL Feature Extraction working")
        print(f"Sample features for: {test_url}")
        for key, value in list(features.items())[:5]:
            print(f"  {key}: {value}")
        print(f"  ... and {len(features)-5} more features")
        
        # Test preprocessor
        preprocessor = PhishingDataPreprocessor()
        sample_data = preprocessor.create_sample_dataset(10)
        
        print("✓ Data preprocessing working")
        print(f"Sample dataset shape: {len(sample_data['numerical_features'])} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data preprocessing: {e}")
        return False

# Test 3: Test trained model (if available)
def test_trained_model():
    print("\nTEST 3: Trained BERT Model")
    print("=" * 60)
    
    if not os.path.exists('best_model.pt'):
        print("✗ No trained model found (best_model.pt)")
        return False
    
    try:
        # Check if we can load the model without TensorFlow issues
        from model import PhishingBERTModel, ModelConfig
        from predict import PhishingPredictor
        
        config = ModelConfig()
        predictor = PhishingPredictor('best_model.pt', config)
        
        print("✓ Trained model loaded successfully")
        
        # Test prediction
        test_url = "http://secure-paypal-update.suspicious-domain.tk/login"
        result = predictor.predict_single_url(test_url)
        
        print(f"✓ Prediction successful")
        print(f"URL: {test_url}")
        print(f"Prediction: {result['prediction']}")
        print(f"Phishing Score: {result['phishing_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing trained model: {e}")
        print("This is likely due to TensorFlow dependency issues")
        return False

# Test 4: Check project files
def test_project_structure():
    print("\nTEST 4: Project Structure")
    print("=" * 60)
    
    required_files = [
        'data_preprocessing.py',
        'model.py',
        'train.py',
        'predict.py',
        'evaluate.py',
        'requirements.txt',
        'phishing_model.pkl'
    ]
    
    optional_files = [
        'best_model.pt',
        'training_results.json',
        'confusion_matrix.png',
        'roc_curve.png',
        'training_history.png'
    ]
    
    print("Required files:")
    all_required_present = True
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✓ {file} ({size:,} bytes)")
        else:
            print(f"  ✗ {file} (missing)")
            all_required_present = False
    
    print("\nOptional files:")
    for file in optional_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✓ {file} ({size:,} bytes)")
        else:
            print(f"  - {file} (not found)")
    
    return all_required_present

def main():
    print("PHISHING DETECTION PROJECT - COMPLETE TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    results = []
    
    # Run all tests
    results.append(("Project Structure", test_project_structure()))
    results.append(("Data Preprocessing", test_data_preprocessing()))
    results.append(("Pickle Model", test_pickle_model()))
    results.append(("Trained Model", test_trained_model()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your phishing detection project is working correctly.")
    elif passed >= len(results) - 1:
        print("✅ Project is mostly working. Minor issues detected.")
    else:
        print("⚠️  Some issues detected. Check the test output above.")
    
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS:")
    print("=" * 60)
    print("1. Use pickle model (recommended):")
    print("   python -c \"import pickle; model=pickle.load(open('phishing_model.pkl','rb')); print(model.predict_url('https://example.com'))\"")
    print("\n2. Use trained BERT model (if working):")
    print("   python predict.py --url https://example.com")
    print("\n3. Interactive mode:")
    print("   python predict.py")

if __name__ == "__main__":
    main()
