import torch
import numpy as np
import argparse
import json
from typing import List, Dict, Union
from datetime import datetime

from model import PhishingBERTModel, ModelConfig
from data_preprocessing import PhishingDataPreprocessor

class PhishingPredictor:
    """Real-time phishing URL prediction class"""
    
    def __init__(self, model_path: str, config: ModelConfig = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or ModelConfig()
        
        # Initialize preprocessor
        self.preprocessor = PhishingDataPreprocessor(
            model_name=self.config.model_name,
            max_length=self.config.max_length
        )
        
        # Load model
        self.model = PhishingBERTModel(
            model_name=self.config.model_name,
            num_numerical_features=self.config.num_numerical_features,
            hidden_size=self.config.hidden_size,
            dropout_rate=self.config.dropout_rate,
            num_classes=self.config.num_classes
        ).to(self.device)
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict_single_url(self, url: str) -> Dict[str, Union[str, float, Dict]]:
        """Predict if a single URL is phishing or legitimate"""
        # Preprocess the URL
        data = self.preprocessor.preprocess_urls([url])
        
        # Move to device
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        numerical_features = data['numerical_features'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, numerical_features)
            
            prediction = torch.argmax(outputs['logits'], dim=1).item()
            probabilities = outputs['probabilities'][0].cpu().numpy()
            phishing_score = probabilities[1]
        
        # Determine risk level
        if phishing_score >= 0.8:
            risk_level = "HIGH"
        elif phishing_score >= 0.6:
            risk_level = "MEDIUM"
        elif phishing_score >= 0.4:
            risk_level = "LOW"
        else:
            risk_level = "VERY_LOW"
        
        # Extract features for explanation
        feature_extractor = self.preprocessor.feature_extractor
        url_features = feature_extractor.extract_url_features(url)
        
        # Identify suspicious features
        suspicious_features = []
        if url_features['has_ip']:
            suspicious_features.append("Contains IP address instead of domain")
        if url_features['suspicious_keywords_count'] > 2:
            suspicious_features.append(f"Contains {int(url_features['suspicious_keywords_count'])} suspicious keywords")
        if url_features['url_length'] > 100:
            suspicious_features.append("Unusually long URL")
        if url_features['subdomain_count'] > 3:
            suspicious_features.append("Multiple subdomains")
        if url_features['has_suspicious_tld']:
            suspicious_features.append("Suspicious top-level domain")
        if not url_features['is_https']:
            suspicious_features.append("Not using HTTPS")
        if url_features['domain_entropy'] > 4.5:
            suspicious_features.append("High domain entropy (random-looking)")
        
        return {
            'url': url,
            'prediction': 'PHISHING' if prediction == 1 else 'LEGITIMATE',
            'phishing_score': float(phishing_score),
            'legitimate_score': float(probabilities[0]),
            'risk_level': risk_level,
            'confidence': float(max(probabilities)),
            'suspicious_features': suspicious_features,
            'url_features': {k: float(v) for k, v in url_features.items()},
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, urls: List[str]) -> List[Dict[str, Union[str, float, Dict]]]:
        """Predict multiple URLs at once"""
        results = []
        
        # Process in batches for memory efficiency
        batch_size = 32
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            
            # Preprocess batch
            data = self.preprocessor.preprocess_urls(batch_urls)
            
            # Move to device
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            numerical_features = data['numerical_features'].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, numerical_features)
                
                predictions = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
                probabilities = outputs['probabilities'].cpu().numpy()
            
            # Process results
            for j, url in enumerate(batch_urls):
                prediction = predictions[j]
                probs = probabilities[j]
                phishing_score = probs[1]
                
                # Determine risk level
                if phishing_score >= 0.8:
                    risk_level = "HIGH"
                elif phishing_score >= 0.6:
                    risk_level = "MEDIUM"
                elif phishing_score >= 0.4:
                    risk_level = "LOW"
                else:
                    risk_level = "VERY_LOW"
                
                results.append({
                    'url': url,
                    'prediction': 'PHISHING' if prediction == 1 else 'LEGITIMATE',
                    'phishing_score': float(phishing_score),
                    'legitimate_score': float(probs[0]),
                    'risk_level': risk_level,
                    'confidence': float(max(probs))
                })
        
        return results
    
    def analyze_url_features(self, url: str) -> Dict[str, Union[str, float, List]]:
        """Detailed analysis of URL features"""
        feature_extractor = self.preprocessor.feature_extractor
        features = feature_extractor.extract_url_features(url)
        
        # Categorize features
        length_features = {
            'url_length': features['url_length'],
            'domain_length': features['domain_length'],
            'path_length': features['path_length']
        }
        
        character_features = {
            'num_dots': features['num_dots'],
            'num_hyphens': features['num_hyphens'],
            'num_underscores': features['num_underscores'],
            'num_slashes': features['num_slashes'],
            'digit_ratio': features['digit_ratio'],
            'special_char_ratio': features['special_char_ratio']
        }
        
        security_features = {
            'is_https': bool(features['is_https']),
            'has_ip': bool(features['has_ip']),
            'has_port': bool(features['has_port']),
            'has_suspicious_tld': bool(features['has_suspicious_tld'])
        }
        
        complexity_features = {
            'domain_entropy': features['domain_entropy'],
            'url_entropy': features['url_entropy'],
            'subdomain_count': features['subdomain_count'],
            'suspicious_keywords_count': features['suspicious_keywords_count']
        }
        
        # Generate warnings
        warnings = []
        if features['url_length'] > 100:
            warnings.append(f"Very long URL ({features['url_length']} characters)")
        if features['has_ip']:
            warnings.append("Uses IP address instead of domain name")
        if not features['is_https']:
            warnings.append("Not using secure HTTPS protocol")
        if features['suspicious_keywords_count'] > 0:
            warnings.append(f"Contains {features['suspicious_keywords_count']} suspicious keywords")
        if features['subdomain_count'] > 3:
            warnings.append(f"Has {features['subdomain_count']} subdomains (suspicious)")
        if features['domain_entropy'] > 4.5:
            warnings.append("Domain appears randomly generated")
        
        return {
            'url': url,
            'length_features': length_features,
            'character_features': character_features,
            'security_features': security_features,
            'complexity_features': complexity_features,
            'warnings': warnings,
            'overall_suspicion_score': sum([
                features['suspicious_keywords_count'] * 0.2,
                features['has_ip'] * 0.3,
                (not features['is_https']) * 0.2,
                min(features['subdomain_count'] / 5, 1) * 0.15,
                min(features['domain_entropy'] / 6, 1) * 0.15
            ])
        }

def main():
    """Command-line interface for phishing prediction"""
    parser = argparse.ArgumentParser(description='Phishing URL Detection using BERT')
    parser.add_argument('--url', type=str, help='Single URL to analyze')
    parser.add_argument('--file', type=str, help='File containing URLs (one per line)')
    parser.add_argument('--model', type=str, default='best_model.pt', help='Path to trained model')
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--detailed', action='store_true', help='Include detailed feature analysis')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Initialize predictor
    config = ModelConfig()
    predictor = PhishingPredictor(args.model, config)
    
    results = []
    
    if args.url:
        # Single URL prediction
        print(f"Analyzing URL: {args.url}")
        result = predictor.predict_single_url(args.url)
        results.append(result)
        
        # Print results
        print(f"\nPrediction: {result['prediction']}")
        print(f"Phishing Score: {result['phishing_score']:.4f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if result['suspicious_features']:
            print("\nSuspicious Features:")
            for feature in result['suspicious_features']:
                print(f"  - {feature}")
        
        if args.detailed:
            analysis = predictor.analyze_url_features(args.url)
            print(f"\nDetailed Analysis:")
            print(f"Overall Suspicion Score: {analysis['overall_suspicion_score']:.4f}")
            if analysis['warnings']:
                print("Warnings:")
                for warning in analysis['warnings']:
                    print(f"  - {warning}")
    
    elif args.file:
        # Batch prediction from file
        try:
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(urls)} URLs from {args.file}")
            results = predictor.predict_batch(urls)
            
            # Print summary
            phishing_count = sum(1 for r in results if r['prediction'] == 'PHISHING')
            print(f"\nResults Summary:")
            print(f"Total URLs: {len(results)}")
            print(f"Phishing URLs: {phishing_count}")
            print(f"Legitimate URLs: {len(results) - phishing_count}")
            
            # Show high-risk URLs
            high_risk = [r for r in results if r['risk_level'] == 'HIGH']
            if high_risk:
                print(f"\nHigh-Risk URLs ({len(high_risk)}):")
                for result in high_risk[:10]:  # Show top 10
                    print(f"  {result['url']} (Score: {result['phishing_score']:.4f})")
        
        except FileNotFoundError:
            print(f"Error: File {args.file} not found")
            return
    
    else:
        # Interactive mode
        print("Interactive Phishing URL Detector")
        print("Enter URLs to analyze (type 'quit' to exit):")
        
        while True:
            url = input("\nURL: ").strip()
            if url.lower() in ['quit', 'exit', 'q']:
                break
            
            if not url:
                continue
            
            try:
                result = predictor.predict_single_url(url)
                results.append(result)
                
                print(f"Prediction: {result['prediction']}")
                print(f"Phishing Score: {result['phishing_score']:.4f}")
                print(f"Risk Level: {result['risk_level']}")
                
                if result['suspicious_features']:
                    print("Suspicious Features:")
                    for feature in result['suspicious_features']:
                        print(f"  - {feature}")
            
            except Exception as e:
                print(f"Error analyzing URL: {e}")
    
    # Save results if output file specified
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
