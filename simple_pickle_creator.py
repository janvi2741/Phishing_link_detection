import pickle
import torch
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from urllib.parse import urlparse
import re

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.suspicious_keywords = [
            'secure', 'account', 'update', 'confirm', 'verify', 'login',
            'signin', 'banking', 'paypal', 'amazon', 'microsoft', 'apple',
            'google', 'facebook', 'twitter', 'instagram', 'suspended',
            'limited', 'locked', 'expired', 'urgent', 'immediate'
        ]
    
    def extract_url_features(self, url: str) -> Dict[str, float]:
        """Extract numerical features from URL"""
        parsed = urlparse(url)
        
        features = {
            'url_length': len(url),
            'domain_length': len(parsed.netloc),
            'path_length': len(parsed.path),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_question_marks': url.count('?'),
            'num_equal_signs': url.count('='),
            'num_ampersands': url.count('&'),
            'has_ip': self._has_ip_address(parsed.netloc),
            'has_port': 1 if parsed.port else 0,
            'is_https': 1 if parsed.scheme == 'https' else 0,
            'suspicious_keywords_count': self._count_suspicious_keywords(url.lower()),
            'domain_entropy': self._calculate_entropy(parsed.netloc),
            'url_entropy': self._calculate_entropy(url),
            'subdomain_count': len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0,
            'has_suspicious_tld': self._has_suspicious_tld(parsed.netloc),
            'digit_ratio': sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
            'special_char_ratio': sum(not c.isalnum() and c not in '.-_/' for c in url) / len(url) if len(url) > 0 else 0
        }
        
        return features
    
    def _has_ip_address(self, domain: str) -> float:
        """Check if domain is an IP address"""
        ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        return 1.0 if re.match(ip_pattern, domain) else 0.0
    
    def _count_suspicious_keywords(self, url: str) -> float:
        """Count suspicious keywords in URL"""
        return sum(1 for keyword in self.suspicious_keywords if keyword in url)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        text_len = len(text)
        for count in char_counts.values():
            prob = count / text_len
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _has_suspicious_tld(self, domain: str) -> float:
        """Check for suspicious top-level domains"""
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download', '.stream']
        return 1.0 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0.0

class SimplePhishingPredictor:
    """Simplified phishing predictor using rule-based approach"""
    
    def __init__(self):
        self.feature_extractor = URLFeatureExtractor()
        self.creation_date = datetime.now().isoformat()
        
    def predict_url(self, url: str) -> Dict[str, Any]:
        """Predict if a URL is phishing using rule-based approach"""
        features = self.feature_extractor.extract_url_features(url)
        
        # Rule-based scoring
        phishing_score = 0.0
        
        # URL length penalty
        if features['url_length'] > 100:
            phishing_score += 0.2
        elif features['url_length'] > 75:
            phishing_score += 0.1
            
        # Suspicious keywords
        phishing_score += min(features['suspicious_keywords_count'] * 0.15, 0.4)
        
        # IP address instead of domain
        if features['has_ip']:
            phishing_score += 0.3
            
        # Not HTTPS
        if not features['is_https']:
            phishing_score += 0.15
            
        # Suspicious TLD
        if features['has_suspicious_tld']:
            phishing_score += 0.25
            
        # High entropy (random-looking domain)
        if features['domain_entropy'] > 4.5:
            phishing_score += 0.2
        elif features['domain_entropy'] > 4.0:
            phishing_score += 0.1
            
        # Multiple subdomains
        if features['subdomain_count'] > 3:
            phishing_score += 0.15
        elif features['subdomain_count'] > 2:
            phishing_score += 0.1
            
        # High special character ratio
        if features['special_char_ratio'] > 0.3:
            phishing_score += 0.1
            
        # Normalize score to [0, 1]
        phishing_score = min(phishing_score, 1.0)
        legitimate_score = 1.0 - phishing_score
        
        # Determine prediction and risk level
        prediction = 'PHISHING' if phishing_score > 0.5 else 'LEGITIMATE'
        
        if phishing_score >= 0.8:
            risk_level = "HIGH"
        elif phishing_score >= 0.6:
            risk_level = "MEDIUM"
        elif phishing_score >= 0.4:
            risk_level = "LOW"
        else:
            risk_level = "VERY_LOW"
            
        # Identify suspicious features
        suspicious_features = []
        if features['has_ip']:
            suspicious_features.append("Contains IP address instead of domain")
        if features['suspicious_keywords_count'] > 2:
            suspicious_features.append(f"Contains {int(features['suspicious_keywords_count'])} suspicious keywords")
        if features['url_length'] > 100:
            suspicious_features.append("Unusually long URL")
        if features['subdomain_count'] > 3:
            suspicious_features.append("Multiple subdomains")
        if features['has_suspicious_tld']:
            suspicious_features.append("Suspicious top-level domain")
        if not features['is_https']:
            suspicious_features.append("Not using HTTPS")
        if features['domain_entropy'] > 4.5:
            suspicious_features.append("High domain entropy (random-looking)")
            
        return {
            'url': url,
            'prediction': prediction,
            'phishing_score': float(phishing_score),
            'legitimate_score': float(legitimate_score),
            'risk_level': risk_level,
            'confidence': float(max(phishing_score, legitimate_score)),
            'suspicious_features': suspicious_features,
            'url_features': {k: float(v) for k, v in features.items()},
            'timestamp': datetime.now().isoformat(),
            'model_type': 'rule_based'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            'model_type': 'rule_based',
            'creation_date': self.creation_date,
            'features_count': 20,
            'description': 'Rule-based phishing detection using URL features'
        }

def create_simple_pickle(output_path: str = 'phishing_model.pkl'):
    """Create a simple pickle file with rule-based predictor"""
    
    print("Creating simple phishing detection model...")
    
    # Create the predictor
    predictor = SimplePhishingPredictor()
    
    try:
        # Save to pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(predictor, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model saved successfully to {output_path}")
        
        # Get file size
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"Pickle file size: {file_size:.1f} KB")
        
        # Test the model
        test_urls = [
            "https://www.google.com",
            "http://secure-paypal-update.suspicious-domain.tk/login",
            "https://github.com/user/repository",
            "https://amazon-security-alert.fake-site.ml/verify"
        ]
        
        print("\nTesting predictions:")
        for url in test_urls:
            result = predictor.predict_url(url)
            print(f"   {url}")
            print(f"   -> {result['prediction']} (Score: {result['phishing_score']:.3f}, Risk: {result['risk_level']})")
        
        print(f"\nPickle file created successfully!")
        print(f"File location: {os.path.abspath(output_path)}")
        
        return output_path
        
    except Exception as e:
        print(f"Error creating pickle file: {e}")
        return None

if __name__ == "__main__":
    create_simple_pickle()
