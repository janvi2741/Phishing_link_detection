import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
import os
import json
from datetime import datetime

from model import PhishingBERTModel, PhishingLoss, ModelConfig
from data_preprocessing import PhishingDataPreprocessor

class PhishingTrainer:
    """Trainer class for phishing detection model"""
    
    def __init__(self, config: ModelConfig, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = PhishingBERTModel(
            model_name=config.model_name,
            num_numerical_features=config.num_numerical_features,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate,
            num_classes=config.num_classes
        ).to(self.device)
        
        # Initialize loss function
        self.criterion = PhishingLoss(
            class_weights=config.class_weights.to(self.device),
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Model initialized on device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def create_data_loaders(self, data: dict, train_split: float = 0.8, val_split: float = 0.1):
        """Create train, validation, and test data loaders"""
        
        # Create dataset
        dataset = TensorDataset(
            data['input_ids'],
            data['attention_mask'],
            data['numerical_features'],
            data['labels']
        )
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids, attention_mask, numerical_features, labels = [b.to(self.device) for b in batch]
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, numerical_features)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> tuple:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids, attention_mask, numerical_features, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(input_ids, attention_mask, numerical_features)
                loss = self.criterion(outputs['logits'], labels)
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0
        patience = 3
        patience_counter = 0
        
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.save_model('best_model.pt')
                patience_counter = 0
                print(f"New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate model on test set"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids, attention_mask, numerical_features, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(input_ids, attention_mask, numerical_features)
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = outputs['probabilities']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        auc_score = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=['Legitimate', 'Phishing'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, labels: list, probabilities: list):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, np.array(probabilities)[:, 1])
        auc_score = roc_auc_score(labels, np.array(probabilities)[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Model loaded from {filename}")

def main():
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize configuration
    config = ModelConfig()
    
    # Create data preprocessor
    preprocessor = PhishingDataPreprocessor(
        model_name=config.model_name,
        max_length=config.max_length
    )
    
    # Create sample dataset (replace with your actual data)
    print("Creating sample dataset...")
    data = preprocessor.create_sample_dataset(num_samples=2000)
    
    # Initialize trainer
    trainer = PhishingTrainer(config)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = trainer.create_data_loaders(data)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Load best model for evaluation
    trainer.load_model('best_model.pt')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_loader)
    
    # Print results
    print(f"\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC Score: {results['auc_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        results['labels'], 
        results['predictions'],
        target_names=['Legitimate', 'Phishing']
    ))
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(results['confusion_matrix'])
    trainer.plot_roc_curve(results['labels'], results['probabilities'])
    
    # Save final results
    with open('training_results.json', 'w') as f:
        results_to_save = {
            'accuracy': results['accuracy'],
            'auc_score': results['auc_score'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'timestamp': datetime.now().isoformat()
        }
        json.dump(results_to_save, f, indent=2)
    
    print("\nTraining completed successfully!")
    print("Files saved:")
    print("- best_model.pt (model checkpoint)")
    print("- training_history.png (training curves)")
    print("- confusion_matrix.png (confusion matrix)")
    print("- roc_curve.png (ROC curve)")
    print("- training_results.json (evaluation results)")

if __name__ == "__main__":
    main()
