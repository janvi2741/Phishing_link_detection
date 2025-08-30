import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from typing import Dict, List, Tuple
import json
from datetime import datetime

from model import PhishingBERTModel, ModelConfig
from data_preprocessing import PhishingDataPreprocessor

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path: str, config: ModelConfig, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Load model
        self.model = PhishingBERTModel(
            model_name=config.model_name,
            num_numerical_features=config.num_numerical_features,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate,
            num_classes=config.num_classes
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, urls: List[str]) -> Dict[str, np.ndarray]:
        """Make predictions on a list of URLs"""
        preprocessor = PhishingDataPreprocessor(
            model_name=self.config.model_name,
            max_length=self.config.max_length
        )
        
        # Preprocess URLs
        data = preprocessor.preprocess_urls(urls)
        
        # Move to device
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        numerical_features = data['numerical_features'].to(self.device)
        
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            # Process in batches
            batch_size = 32
            for i in range(0, len(urls), batch_size):
                batch_input_ids = input_ids[i:i+batch_size]
                batch_attention_mask = attention_mask[i:i+batch_size]
                batch_numerical_features = numerical_features[i:i+batch_size]
                
                outputs = self.model(batch_input_ids, batch_attention_mask, batch_numerical_features)
                
                batch_predictions = torch.argmax(outputs['logits'], dim=1)
                batch_probabilities = outputs['probabilities']
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probabilities.cpu().numpy())
        
        return {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'phishing_scores': np.array(probabilities)[:, 1]  # Probability of being phishing
        }
    
    def evaluate_dataset(self, urls: List[str], labels: List[int]) -> Dict:
        """Evaluate model on a dataset"""
        results = self.predict(urls)
        predictions = results['predictions']
        probabilities = results['probabilities']
        phishing_scores = results['phishing_scores']
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1_score': f1_score(labels, predictions),
            'auc_roc': roc_auc_score(labels, phishing_scores),
            'auc_pr': average_precision_score(labels, phishing_scores)
        }
        
        # Detailed classification report
        class_report = classification_report(
            labels, predictions,
            target_names=['Legitimate', 'Phishing'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        return {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'phishing_scores': phishing_scores,
            'labels': labels
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = 'confusion_matrix_eval.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
            annotations.append(row)
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'],
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, labels: List[int], scores: np.ndarray, save_path: str = 'roc_curve_eval.png'):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_score = roc_auc_score(labels, scores)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, labels: List[int], scores: np.ndarray, 
                                   save_path: str = 'pr_curve_eval.png'):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(labels, scores)
        auc_pr = average_precision_score(labels, scores)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=3,
                label=f'PR curve (AUC = {auc_pr:.3f})')
        
        # Baseline (random classifier)
        baseline = sum(labels) / len(labels)
        plt.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                   label=f'Random classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_distribution(self, labels: List[int], scores: np.ndarray,
                               save_path: str = 'score_distribution.png'):
        """Plot distribution of phishing scores"""
        legitimate_scores = scores[np.array(labels) == 0]
        phishing_scores = scores[np.array(labels) == 1]
        
        plt.figure(figsize=(12, 8))
        
        plt.hist(legitimate_scores, bins=50, alpha=0.7, label='Legitimate URLs', 
                color='blue', density=True)
        plt.hist(phishing_scores, bins=50, alpha=0.7, label='Phishing URLs', 
                color='red', density=True)
        
        plt.xlabel('Phishing Score', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Distribution of Phishing Scores', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassifications(self, urls: List[str], labels: List[int], 
                                 predictions: np.ndarray, scores: np.ndarray) -> Dict:
        """Analyze misclassified examples"""
        misclassified_indices = np.where(np.array(labels) != predictions)[0]
        
        false_positives = []  # Legitimate URLs classified as phishing
        false_negatives = []  # Phishing URLs classified as legitimate
        
        for idx in misclassified_indices:
            example = {
                'url': urls[idx],
                'true_label': 'Phishing' if labels[idx] == 1 else 'Legitimate',
                'predicted_label': 'Phishing' if predictions[idx] == 1 else 'Legitimate',
                'phishing_score': scores[idx],
                'index': idx
            }
            
            if labels[idx] == 0 and predictions[idx] == 1:  # False positive
                false_positives.append(example)
            elif labels[idx] == 1 and predictions[idx] == 0:  # False negative
                false_negatives.append(example)
        
        # Sort by confidence (distance from 0.5)
        false_positives.sort(key=lambda x: x['phishing_score'], reverse=True)
        false_negatives.sort(key=lambda x: x['phishing_score'])
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_misclassified': len(misclassified_indices),
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives)
        }
    
    def generate_evaluation_report(self, urls: List[str], labels: List[int], 
                                 save_path: str = 'evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        print("Evaluating model...")
        results = self.evaluate_dataset(urls, labels)
        
        print("Analyzing misclassifications...")
        misclassification_analysis = self.analyze_misclassifications(
            urls, labels, results['predictions'], results['phishing_scores']
        )
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(urls),
                'legitimate_count': sum(1 for l in labels if l == 0),
                'phishing_count': sum(1 for l in labels if l == 1),
                'class_distribution': {
                    'legitimate_ratio': sum(1 for l in labels if l == 0) / len(labels),
                    'phishing_ratio': sum(1 for l in labels if l == 1) / len(labels)
                }
            },
            'performance_metrics': results['metrics'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'misclassification_analysis': {
                'total_misclassified': misclassification_analysis['total_misclassified'],
                'false_positive_count': misclassification_analysis['fp_count'],
                'false_negative_count': misclassification_analysis['fn_count'],
                'top_false_positives': misclassification_analysis['false_positives'][:10],
                'top_false_negatives': misclassification_analysis['false_negatives'][:10]
            }
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {save_path}")
        return report
    
    def create_evaluation_plots(self, urls: List[str], labels: List[int]):
        """Create all evaluation plots"""
        results = self.evaluate_dataset(urls, labels)
        
        print("Creating evaluation plots...")
        
        # Confusion Matrix
        self.plot_confusion_matrix(results['confusion_matrix'])
        
        # ROC Curve
        self.plot_roc_curve(labels, results['phishing_scores'])
        
        # Precision-Recall Curve
        self.plot_precision_recall_curve(labels, results['phishing_scores'])
        
        # Score Distribution
        self.plot_score_distribution(labels, results['phishing_scores'])
        
        print("All evaluation plots created successfully!")

def main():
    """Main evaluation function"""
    # Initialize configuration
    config = ModelConfig()
    
    # Check if model exists
    model_path = 'best_model.pt'
    try:
        evaluator = ModelEvaluator(model_path, config)
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Create test dataset
    preprocessor = PhishingDataPreprocessor(
        model_name=config.model_name,
        max_length=config.max_length
    )
    
    print("Creating test dataset...")
    test_data = preprocessor.create_sample_dataset(num_samples=500)
    
    # Convert to lists for evaluation
    urls = []
    labels = test_data['labels'].numpy().tolist()
    
    # Generate sample URLs for demonstration
    legitimate_urls = [
        "https://www.google.com/search?q=python",
        "https://github.com/user/repository",
        "https://stackoverflow.com/questions/12345",
        "https://www.wikipedia.org/wiki/Machine_Learning",
        "https://docs.python.org/3/library/urllib.html"
    ]
    
    phishing_urls = [
        "http://secure-paypal-update.suspicious-domain.tk/login",
        "https://amazon-security-alert.fake-site.ml/verify",
        "http://microsoft-account-suspended.phishing.ga/confirm",
        "https://google-security-check.malicious.cf/update",
        "http://facebook-account-locked.fake.click/signin"
    ]
    
    # Create URL list matching the labels
    for i, label in enumerate(labels):
        if label == 0:  # Legitimate
            urls.append(legitimate_urls[i % len(legitimate_urls)])
        else:  # Phishing
            urls.append(phishing_urls[i % len(phishing_urls)])
    
    # Generate comprehensive evaluation report
    report = evaluator.generate_evaluation_report(urls, labels)
    
    # Create evaluation plots
    evaluator.create_evaluation_plots(urls, labels)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(urls)}")
    print(f"Accuracy: {report['performance_metrics']['accuracy']:.4f}")
    print(f"Precision: {report['performance_metrics']['precision']:.4f}")
    print(f"Recall: {report['performance_metrics']['recall']:.4f}")
    print(f"F1-Score: {report['performance_metrics']['f1_score']:.4f}")
    print(f"AUC-ROC: {report['performance_metrics']['auc_roc']:.4f}")
    print(f"AUC-PR: {report['performance_metrics']['auc_pr']:.4f}")
    print("\nFiles generated:")
    print("- evaluation_report.json")
    print("- confusion_matrix_eval.png")
    print("- roc_curve_eval.png")
    print("- pr_curve_eval.png")
    print("- score_distribution.png")

if __name__ == "__main__":
    main()
