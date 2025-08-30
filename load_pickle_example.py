import pickle
import os
from typing import Dict, Any, List
from create_final_pickle import PhishingPredictor

def load_phishing_model(pickle_path: str = 'phishing_model.pkl'):
    """Load the pickled phishing detection model"""
    try:
        with open(pickle_path, 'rb') as f:
            model_package = pickle.load(f)
        print(f"Model loaded successfully from {pickle_path}")
        return model_package
    except FileNotFoundError:
        print(f"Pickle file not found: {pickle_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_single_url(model_package, url: str) -> Dict[str, Any]:
    """Predict if a single URL is phishing or legitimate"""
    try:
        result = model_package.predict_url(url)
        return result
    except Exception as e:
        print(f"Error predicting URL: {e}")
        return None

def predict_multiple_urls(model_package, urls: List[str]) -> List[Dict[str, Any]]:
    """Predict multiple URLs"""
    results = []
    for url in urls:
        result = predict_single_url(model_package, url)
        if result:
            results.append(result)
    return results

def display_prediction_result(result: Dict[str, Any]):
    """Display prediction result in a formatted way"""
    if not result:
        return
    
    print(f"\nURL: {result['url']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Phishing Score: {result['phishing_score']:.4f}")
    print(f"Legitimate Score: {result['legitimate_score']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    if result['suspicious_features']:
        print("Suspicious Features:")
        for feature in result['suspicious_features']:
            print(f"   - {feature}")
    else:
        print("No suspicious features detected")

def interactive_mode(model_package):
    """Interactive mode for URL analysis"""
    print("\nInteractive Phishing URL Detector")
    print("Enter URLs to analyze (type 'quit' to exit):")
    
    while True:
        url = input("\nEnter URL: ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not url:
            continue
        
        result = predict_single_url(model_package, url)
        if result:
            display_prediction_result(result)

def batch_analysis_from_file(model_package, file_path: str):
    """Analyze URLs from a text file"""
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(urls)} URLs from {file_path}")
        results = predict_multiple_urls(model_package, urls)
        
        # Summary statistics
        phishing_count = sum(1 for r in results if r['prediction'] == 'PHISHING')
        high_risk_count = sum(1 for r in results if r['risk_level'] == 'HIGH')
        
        print(f"\nAnalysis Summary:")
        print(f"   Total URLs: {len(results)}")
        print(f"   Phishing URLs: {phishing_count}")
        print(f"   Legitimate URLs: {len(results) - phishing_count}")
        print(f"   High Risk URLs: {high_risk_count}")
        
        # Show high-risk URLs
        high_risk_urls = [r for r in results if r['risk_level'] == 'HIGH']
        if high_risk_urls:
            print(f"\nHigh-Risk URLs:")
            for result in high_risk_urls[:10]:  # Show top 10
                print(f"   - {result['url']} (Score: {result['phishing_score']:.4f})")
        
        return results
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def main():
    """Main function demonstrating how to use the pickled model"""
    
    # Load the model
    model_package = load_phishing_model('phishing_model.pkl')
    
    if not model_package:
        print("Could not load model. Make sure 'phishing_model.pkl' exists.")
        return
    
    # Display model information
    info = model_package.get_model_info()
    print(f"\nModel Information:")
    print(f"   Creation Date: {info['metadata']['creation_date']}")
    print(f"   Model Loaded: {info['metadata']['model_loaded']}")
    print(f"   Total Parameters: {info['model_parameters']['total_parameters']:,}")
    
    # Example 1: Single URL prediction
    print(f"\nExample 1: Single URL Analysis")
    test_url = "https://secure-paypal-update.suspicious-domain.tk/login"
    result = predict_single_url(model_package, test_url)
    if result:
        display_prediction_result(result)
    
    # Example 2: Multiple URLs
    print(f"\nExample 2: Multiple URL Analysis")
    test_urls = [
        "https://www.google.com",
        "https://github.com/user/repository",
        "http://amazon-security-alert.fake-site.ml/verify",
        "https://www.wikipedia.org/wiki/Machine_Learning"
    ]
    
    results = predict_multiple_urls(model_package, test_urls)
    for result in results:
        display_prediction_result(result)
    
    # Example 3: Interactive mode (commented out for demo)
    # interactive_mode(model_package)
    
    print(f"\nDemo completed! The model is ready to use.")

if __name__ == "__main__":
    main()
