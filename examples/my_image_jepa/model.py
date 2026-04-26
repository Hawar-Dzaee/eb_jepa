import torch.nn as nn 
import torchvision


class ResNet18(nn.Module):
    def __init__(self): 
        super().__init__()
        self.backbone = torchvision.models.resnet18()
        self.backbone.fc = nn.Identity()
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,stride=1,padding=2,bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.features_dim = 512 

    def forward(self,x): 
        return self.backbone(x)
    

class ImageSSL(nn.Module): 

    def __init__(self,backbone,features_dim,proj_hidden_dim=2048,proj_output_dim=2048): 
        super().__init__()
        self.backbone = backbone  
        self.features_dim = features_dim

        self.projector = nn.Sequential(
            nn.Linear(features_dim,proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim,proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim,proj_output_dim),
        )

    def forward(self,x): 
        features = self.backbone(x) 
        projections = self.projector(features)
        return features,projections
    

class LinearProbe(nn.Module): 
    """Linear probe classifier for evaluation representations."""
    def __init__(self,feature_dim,num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim,num_classes)

    def forward(self,x): 
        return self.classifier(x)
    
