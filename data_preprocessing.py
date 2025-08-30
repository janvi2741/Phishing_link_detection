import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from typing import List, Tuple, Dict, Optional
import warnings

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some functionality will be limited.")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. BERT functionality will be disabled.")

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

class PhishingDataPreprocessor:
    """Preprocess data for BERT-based phishing detection"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.feature_extractor = URLFeatureExtractor()
        
        # Initialize tokenizer only if transformers is available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert_enabled = True
            except Exception as e:
                warnings.warn(f"Failed to load tokenizer: {e}. BERT functionality disabled.")
                self.tokenizer = None
                self.bert_enabled = False
        else:
            self.tokenizer = None
            self.bert_enabled = False
    
    def preprocess_urls(self, urls: List[str], labels: Optional[List[int]] = None) -> Dict[str, any]:
        """Preprocess URLs for model training/inference"""
        # Extract numerical features (always available)
        numerical_features = []
        for url in urls:
            features = self.feature_extractor.extract_url_features(url)
            numerical_features.append(list(features.values()))
        
        result = {
            'numerical_features': numerical_features
        }
        
        # Add BERT features if available
        if self.bert_enabled and self.tokenizer is not None:
            # Clean and prepare URLs for tokenization
            processed_urls = []
            for url in urls:
                # Replace special characters with spaces for better tokenization
                processed_url = re.sub(r'[^\w\s.-]', ' ', url)
                processed_url = re.sub(r'\s+', ' ', processed_url).strip()
                processed_urls.append(processed_url)
            
            try:
                # Tokenize URLs
                encoding = self.tokenizer(
                    processed_urls,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt' if TORCH_AVAILABLE else 'np'
                )
                
                result['input_ids'] = encoding['input_ids']
                result['attention_mask'] = encoding['attention_mask']
            except Exception as e:
                warnings.warn(f"Tokenization failed: {e}. Using only numerical features.")
        
        # Convert to tensors if PyTorch is available
        if TORCH_AVAILABLE:
            result['numerical_features'] = torch.tensor(numerical_features, dtype=torch.float32)
            if labels is not None:
                result['labels'] = torch.tensor(labels, dtype=torch.long)
        else:
            result['numerical_features'] = np.array(numerical_features, dtype=np.float32)
            if labels is not None:
                result['labels'] = np.array(labels, dtype=np.int64)
        
        return result
    
    def create_dataset_from_csv(self, csv_path: str, url_column: str, label_column: str) -> Dict[str, any]:
        """Create dataset from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            urls = df[url_column].tolist()
            labels = df[label_column].tolist()
            
            return self.preprocess_urls(urls, labels)
        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_path}: {e}")
    
    def create_sample_dataset(self, num_samples: int = 1000) -> Dict[str, any]:
        """Create a sample dataset for testing"""
        # Generate sample legitimate URLs
        legitimate_urls = [
            "https://www.google.com/search?q=python",
            "https://github.com/user/repository",
            "https://stackoverflow.com/questions/12345",
            "https://www.wikipedia.org/wiki/Machine_Learning",
            "https://docs.python.org/3/library/urllib.html",
            "https://www.amazon.com/product/12345",
            "https://www.youtube.com/watch?v=abc123",
            "https://www.linkedin.com/in/username",
            "https://www.reddit.com/r/MachineLearning",
            "https://news.ycombinator.com/item?id=12345"
        ]
        
        # Generate sample phishing URLs
        phishing_urls = [
            "http://secure-paypal-update.suspicious-domain.tk/login",
            "https://amazon-security-alert.fake-site.ml/verify",
            "http://microsoft-account-suspended.phishing.ga/confirm",
            "https://google-security-check.malicious.cf/update",
            "http://facebook-account-locked.fake.click/signin",
            "https://apple-id-verification.suspicious.download/login",
            "http://banking-urgent-update.phish.stream/account",
            "https://instagram-verify-now.fake-domain.tk/confirm",
            "http://twitter-suspended-account.malicious.ml/restore",
            "https://linkedin-security-alert.phishing.ga/verify"
        ]
        
        # Create balanced dataset
        urls = []
        labels = []
        
        samples_per_class = num_samples // 2
        
        for i in range(samples_per_class):
            # Add legitimate URL
            urls.append(legitimate_urls[i % len(legitimate_urls)])
            labels.append(0)  # 0 for legitimate
            
            # Add phishing URL
            urls.append(phishing_urls[i % len(phishing_urls)])
            labels.append(1)  # 1 for phishing
        
        return self.preprocess_urls(urls, labels)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        sample_features = self.feature_extractor.extract_url_features("https://example.com")
        return list(sample_features.keys())

if __name__ == "__main__":
    # Example usage
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    
    try:
        preprocessor = PhishingDataPreprocessor()
        print(f"BERT enabled: {preprocessor.bert_enabled}")
        
        # Create sample dataset
        sample_data = preprocessor.create_sample_dataset(100)
        
        print("\nSample dataset created:")
        
        # Show available data
        if 'input_ids' in sample_data:
            if hasattr(sample_data['input_ids'], 'shape'):
                print(f"Input IDs shape: {sample_data['input_ids'].shape}")
            else:
                print(f"Input IDs length: {len(sample_data['input_ids'])}")
        
        if 'attention_mask' in sample_data:
            if hasattr(sample_data['attention_mask'], 'shape'):
                print(f"Attention mask shape: {sample_data['attention_mask'].shape}")
            else:
                print(f"Attention mask length: {len(sample_data['attention_mask'])}")
        
        if hasattr(sample_data['numerical_features'], 'shape'):
            print(f"Numerical features shape: {sample_data['numerical_features'].shape}")
        else:
            print(f"Numerical features shape: {np.array(sample_data['numerical_features']).shape}")
        
        if 'labels' in sample_data:
            if hasattr(sample_data['labels'], 'shape'):
                print(f"Labels shape: {sample_data['labels'].shape}")
            else:
                print(f"Labels length: {len(sample_data['labels'])}")
        
        # Show feature names
        feature_names = preprocessor.get_feature_names()
        print(f"\nFeature names ({len(feature_names)}): {feature_names}")
        
        # Show sample feature extraction
        feature_extractor = URLFeatureExtractor()
        sample_url = "https://secure-paypal-update.suspicious-domain.tk/login"
        features = feature_extractor.extract_url_features(sample_url)
        print(f"\nExtracted features for sample URL:")
        for feature_name, value in features.items():
            print(f"{feature_name}: {value}")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
