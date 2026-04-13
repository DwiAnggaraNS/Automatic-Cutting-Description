import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import timm  # Pastikan sudah terinstall: pip install timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class MultiModelTrainer:
    """
    A unified trainer class for transfer learning with multiple model architectures.
    """
    def __init__(self, model_name, num_classes, device='cuda', pretrained=True):
        """
        Args:
            model_name (str): Identifier for the model (e.g., 'convnext_tiny').
            num_classes (int): Number of output classes (7 for your rock types).
            device (str): 'cuda' or 'cpu'.
            pretrained (bool): Whether to load pretrained weights.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create model and move to device
        self.model = self._create_model(model_name, num_classes, pretrained)
        self.model = self.model.to(self.device)
        
        # Get model-specific transforms and default input size
        self.transforms = self._get_transforms(model_name)
        self.input_size = self._get_default_input_size(model_name)

    def _create_model(self, model_name, num_classes, pretrained):
        """Factory method to create the specified model."""
        if model_name.startswith('convnext'):
            # Using timm for better flexibility
            model = timm.create_model(model_name, pretrained=pretrained)
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes)
            
        elif model_name.startswith('tf_efficientnetv2'):
            # EfficientNetV2 from timm
            model = timm.create_model(model_name, pretrained=pretrained)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            
        elif model_name.startswith('resnet'):
            # ResNet from torchvision
            model = models.__dict__[model_name](pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            
        elif model_name.startswith('densenet'):
            # DenseNet from torchvision
            model = models.__dict__[model_name](pretrained=pretrained)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            
        elif model_name.startswith('davit'):
            # DaViT from timm
            model = timm.create_model(model_name, pretrained=pretrained)
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes)
            
        else:
            raise ValueError(f"Model {model_name} not supported. Add it to the factory.")
            
        return model

    def _get_transforms(self, model_name):
        """Get appropriate data transforms for the model."""
        # Standard ImageNet normalization (works for most torchvision & timm models)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        
        # Base transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        return {'train': train_transform, 'val': val_transform}

    def _get_default_input_size(self, model_name):
        """Return the recommended input size for the model."""
        sizes = {
            'convnext_tiny': 224,
            'tf_efficientnetv2_s': 384,  # Can also use 224 or 288
            'resnet50': 224,
            'densenet121': 224,
            'davit_tiny': 224,
        }
        return sizes.get(model_name, 224)  # Default to 224

    def train_epoch(self, dataloader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        pbar = tqdm(dataloader, desc=f"Training {self.model_name}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1

    def validate_epoch(self, dataloader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validating {self.model_name}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1

    def train(self, train_loader, val_loader, epochs=10, lr=1e-4, 
              loss_fn='cross_entropy', focal_gamma=2.0):
        """
        Main training loop.
        """
        # Setup loss function
        if loss_fn == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'focal':
            # You can implement or import FocalLoss here
            # For simplicity, using CrossEntropy as placeholder
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss function {loss_fn} not supported.")
            
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {'train_loss': [], 'val_loss': [], 
                  'train_acc': [], 'val_acc': [],
                  'train_f1': [], 'val_f1': []}
        
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 30)
            
            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validation
            val_loss, val_acc, val_f1 = self.validate_epoch(
                val_loader, criterion
            )
            
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), f'best_{self.model_name}.pth')
                print(f"Saved best model with Val F1: {best_val_f1:.4f}")
        
        return history

# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    # Assume you have your custom Dataset class (RockDataset) ready
    # train_dataset = RockDataset(root='path/to/train', transform=None)
    # val_dataset = RockDataset(root='path/to/val', transform=None)
    
    # For demonstration, we'll use a dummy dataset
    from torch.utils.data import TensorDataset
    dummy_x = torch.randn(100, 3, 224, 224)
    dummy_y = torch.randint(0, 7, (100,))
    train_dataset = TensorDataset(dummy_x, dummy_y)
    val_dataset = TensorDataset(dummy_x, dummy_y)
    
    # List of models to experiment with
    models_to_train = [
        'convnext_tiny',
        'tf_efficientnetv2_s',
        'resnet50',
        'densenet121',
        'davit_tiny'
    ]
    
    results = {}
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print('='*50)
        
        # Initialize trainer
        trainer = MultiModelTrainer(
            model_name=model_name,
            num_classes=7,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            pretrained=True
        )
        
        # Apply model-specific transforms to dataset
        # (In a real scenario, you'd pass the transform to your Dataset class)
        # train_dataset.transform = trainer.transforms['train']
        # val_dataset.transform = trainer.transforms['val']
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train
        history = trainer.train(
            train_loader, val_loader,
            epochs=10,
            lr=1e-4,
            loss_fn='cross_entropy'  # or 'focal'
        )
        
        results[model_name] = history
        
        # You can also save the model architecture + weights
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'model_name': model_name,
            'num_classes': 7,
            'input_size': trainer.input_size,
        }, f'{model_name}_final.pth')
        
    # After training all models, you can compare their performance
    print("\n" + "="*50)
    print("Final Results Summary")
    print("="*50)
    for model_name, history in results.items():
        best_val_f1 = max(history['val_f1'])
        print(f"{model_name}: Best Val F1 = {best_val_f1:.4f}")