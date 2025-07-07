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





'''
Explainations: 

Conv2d: A layer that scans the image with small windows (filters) to find patterns like edges, spots, etc

- ReLu (Rectified Linear Unit): A simple rule: if a number is negative -> set it to zero. Keeps positive numbers as-is. Mkes the model learn better. 

- MaxPool2d: Reduces the size of the image by keepiung only the most important information (like summarizing). 

- Linear: A classic math formula y = mx + c but with multiple inputs and outputs. 

- forward() function: Deines how the input (image) flows through the layers to produce a prediction. 

- view(): REshapes data. We flatten 3D image features into a 1D list so we can pass it to the fully conneceted layers. 



In pytorch, every custom model you make (a class that inherits from nn.Module) must have a forward() function  or a method. 

It defines how input data (your image) moves through your model, layer by layer, operation by operation, unitl you get the final output. 

You can think of forward() as a blueprint for the model's data path: 
where data goes
what operations happen
how results move from layer to the next. 
'''


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
