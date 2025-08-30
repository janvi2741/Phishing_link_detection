import pickle
import torch
import os
from datetime import datetime
from typing import Dict, Any

from model import PhishingBERTModel, ModelConfig
from data_preprocessing import PhishingDataPreprocessor
from predict import PhishingPredictor

class PhishingModelPackage:
    """Complete package for phishing detection model"""
    
    def __init__(self, model_path: str = 'best_model.pt'):
        self.config = ModelConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize preprocessor
        self.preprocessor = PhishingDataPreprocessor(
            model_name=self.config.model_name,
            max_length=self.config.max_length
        )
        
        # Initialize model
        self.model = PhishingBERTModel(
            model_name=self.config.model_name,
            num_numerical_features=self.config.num_numerical_features,
            hidden_size=self.config.hidden_size,
            dropout_rate=self.config.dropout_rate,
            num_classes=self.config.num_classes
        )
        
        # Load trained weights if available
        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.model_loaded = True
                self.training_history = {
                    'train_losses': checkpoint.get('train_losses', []),
                    'val_losses': checkpoint.get('val_losses', []),
                    'train_accuracies': checkpoint.get('train_accuracies', []),
                    'val_accuracies': checkpoint.get('val_accuracies', [])
                }
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                self.training_history = {}
        else:
            print(f"Warning: Model file {model_path} not found. Creating package with untrained model.")
            self.training_history = {}
        
        # Package metadata
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'num_classes': self.config.num_classes,
            'max_length': self.config.max_length,
            'model_loaded': self.model_loaded,
            'device_used': self.device,
            'pytorch_version': torch.__version__
        }
    
    def predict_url(self, url: str) -> Dict[str, Any]:
        """Predict if a URL is phishing or legitimate"""
        # Create predictor instance
        predictor = PhishingPredictor.__new__(PhishingPredictor)
        predictor.device = 'cpu'  # Use CPU for inference
        predictor.config = self.config
        predictor.preprocessor = self.preprocessor
        predictor.model = self.model.to('cpu')
        predictor.model.eval()
        
        return predictor.predict_single_url(url)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the packaged model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'metadata': self.metadata,
            'model_parameters': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Approximate size in MB
            },
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        return info

def create_pickle_file(model_path: str = 'best_model.pt', output_path: str = 'phishing_model.pkl'):
    """Create a pickle file containing the complete phishing detection model"""
    
    print("Creating phishing detection model package...")
    
    # Create the model package
    model_package = PhishingModelPackage(model_path)
    
    # Save to pickle file
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model package saved successfully to {output_path}")
        
        # Print package information
        info = model_package.get_model_info()
        print(f"\nPackage Information:")
        print(f"   Creation Date: {info['metadata']['creation_date']}")
        print(f"   Model Loaded: {info['metadata']['model_loaded']}")
        print(f"   Total Parameters: {info['model_parameters']['total_parameters']:,}")
        print(f"   Trainable Parameters: {info['model_parameters']['trainable_parameters']:,}")
        print(f"   Approximate Size: {info['model_parameters']['model_size_mb']:.1f} MB")
        print(f"   PyTorch Version: {info['metadata']['pytorch_version']}")
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Pickle File Size: {file_size:.1f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"Error creating pickle file: {e}")
        return None

def load_and_test_pickle(pickle_path: str = 'phishing_model.pkl'):
    """Load and test the pickle file"""
    
    print(f"\nTesting pickle file: {pickle_path}")
    
    try:
        # Load the model package
        with open(pickle_path, 'rb') as f:
            model_package = pickle.load(f)
        
        print("Pickle file loaded successfully!")
        
        # Test with sample URLs
        test_urls = [
            "https://www.google.com",
            "http://secure-paypal-update.suspicious-domain.tk/login",
            "https://github.com/user/repository",
            "https://amazon-security-alert.fake-site.ml/verify"
        ]
        
        print("\nTesting predictions:")
        for url in test_urls:
            try:
                result = model_package.predict_url(url)
                print(f"   {url}")
                print(f"   -> {result['prediction']} (Score: {result['phishing_score']:.3f}, Risk: {result['risk_level']})")
            except Exception as e:
                print(f"   {url}")
                print(f"   -> Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return False

if __name__ == "__main__":
    # Create the pickle file
    pickle_path = create_pickle_file()
    
    if pickle_path:
        # Test the pickle file
        load_and_test_pickle(pickle_path)
        
        print(f"\nPickle file creation completed!")
        print(f"File location: {os.path.abspath(pickle_path)}")
        print(f"\nUsage:")
        print(f"   import pickle")
        print(f"   with open('{pickle_path}', 'rb') as f:")
        print(f"       model = pickle.load(f)")
        print(f"   result = model.predict_url('https://example.com')")
