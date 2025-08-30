import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional

class PhishingBERTModel(nn.Module):
    """BERT-based model for phishing URL detection"""
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 num_numerical_features: int = 20,
                 hidden_size: int = 768,
                 dropout_rate: float = 0.3,
                 num_classes: int = 2):
        super(PhishingBERTModel, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_config = AutoConfig.from_pretrained(model_name)
        
        # Freeze BERT parameters initially (can be unfrozen for fine-tuning)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers for better adaptation
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # Numerical features processing
        self.numerical_fc = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism for BERT output
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for custom layers"""
        for module in [self.numerical_fc, self.fusion_layer, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                numerical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        batch_size = input_ids.size(0)
        
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output and apply attention
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply self-attention to focus on important tokens
        attended_output, attention_weights = self.attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Global average pooling over sequence dimension
        bert_features = torch.mean(attended_output, dim=1)  # [batch_size, hidden_size]
        
        # Process numerical features
        numerical_output = self.numerical_fc(numerical_features)  # [batch_size, 64]
        
        # Fuse BERT and numerical features
        combined_features = torch.cat([bert_features, numerical_output], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'bert_features': bert_features,
            'numerical_features': numerical_output,
            'attention_weights': attention_weights
        }
    
    def unfreeze_bert(self):
        """Unfreeze all BERT parameters for fine-tuning"""
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def freeze_bert(self):
        """Freeze all BERT parameters"""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def get_feature_importance(self, 
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor,
                             numerical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get feature importance scores"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, numerical_features)
            
            # Get attention weights as feature importance for text
            text_importance = outputs['attention_weights'].mean(dim=1)  # Average over heads
            
            # For numerical features, we can use gradient-based importance
            numerical_features.requires_grad_(True)
            logits = self.forward(input_ids, attention_mask, numerical_features)['logits']
            
            # Calculate gradients
            predicted_class = logits.argmax(dim=1)
            grad_outputs = torch.zeros_like(logits)
            grad_outputs.scatter_(1, predicted_class.unsqueeze(1), 1)
            
            gradients = torch.autograd.grad(
                outputs=logits,
                inputs=numerical_features,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False
            )[0]
            
            numerical_importance = torch.abs(gradients)
        
        return {
            'text_attention': text_importance,
            'numerical_importance': numerical_importance
        }

class PhishingLoss(nn.Module):
    """Custom loss function for phishing detection"""
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, 
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super(PhishingLoss, self).__init__()
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss with class weighting
        """
        ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()

class ModelConfig:
    """Configuration class for the phishing detection model"""
    
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.num_numerical_features = 20
        self.hidden_size = 768
        self.dropout_rate = 0.3
        self.num_classes = 2
        self.max_length = 512
        
        # Training parameters
        self.learning_rate = 2e-5
        self.batch_size = 16
        self.num_epochs = 10
        self.warmup_steps = 500
        self.weight_decay = 0.01
        
        # Class weights for imbalanced data
        self.class_weights = torch.tensor([1.0, 1.0])  # Adjust based on data distribution
        
        # Focal loss parameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

if __name__ == "__main__":
    # Test model initialization
    config = ModelConfig()
    model = PhishingBERTModel(
        model_name=config.model_name,
        num_numerical_features=config.num_numerical_features,
        hidden_size=config.hidden_size,
        dropout_rate=config.dropout_rate,
        num_classes=config.num_classes
    )
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length)
    dummy_numerical_features = torch.randn(batch_size, config.num_numerical_features)
    
    outputs = model(dummy_input_ids, dummy_attention_mask, dummy_numerical_features)
    
    print("Model test successful!")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Probabilities shape: {outputs['probabilities'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
