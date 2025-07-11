import torch.nn as nn
import torch.nn.functional as F

# The following is our custom ChestXrayCNN which didn't work properly 

# class ChestXrayCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ChestXrayCNN, self).__init__()

#         # Convolutional Layer 1: input channels=3 (RGB), output=16 filters
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         # Convolutional Layer 2
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         # Convolutional Layer 3
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

#         # Fully connected layers
#         self.fc1 = nn.Linear(64 * 28 * 28, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#         # Max pooling
#         self.pool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         # Convolution + Activation + Pooling
#         x = self.pool(F.relu(self.conv1(x)))  # 224 → 112
#         x = self.pool(F.relu(self.conv2(x)))  # 112 → 56
#         x = self.pool(F.relu(self.conv3(x)))  # 56 → 28

#         # Flatten before fully connected
#         x = x.view(-1, 64 * 28 * 28)

#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x



# def get_model(num_classes=2):
#     return ChestXrayCNN(num_classes=num_classes)




# In order to mitigate the issue; we use resenet18 model which is a pre-trained model that has been trained on a large dataset. 

from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=2):
    # Load the pretrained ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    #Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    #Replacing final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
