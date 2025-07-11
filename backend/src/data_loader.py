import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir, batch_size = 32, img_size = 224, shuffle = True): 

    # Define image transformaton : Resize, Convert to Tensor, Normalize Pixel Values

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize all the images to 224 * 224
        transforms.ToTensor(),  # Convert images to Tensor
        transforms.Normalize([0.5], [0.5]) # Normalize to a range of (-1, 1)

    ])

    # Load images with the labels from folders
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Create a DataLoader to Load the images in batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, dataset.classes # Return class names too


