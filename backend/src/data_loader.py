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



    '''
    Some Explainations: 

    .transforms.Resize((img_size, img_size)) : Images can be of any size. We make them all the same size so the model can handle them.

    .transforms.ToTensor(): Converts images form regular pictures (pixels 0-225) to tensors - which are like multi-dimensioinal lists of numbers. 
    

    .transforms.Normalize([o.5], [0.5]): Shifts all numbers from 0-1 to -1 to 1. This makes training easier for the model (like centering numbers around zero). 


    .datasets.ImageFolder: Looks from images inside the folders like NORMAL/, and PNEUMONIA/ and automatically labels them. 

    .DataLoader: Takes this dataset and feeds it to the model in small batches (chunks) of 32 images at a time - which speeds up and stabilizes learning.
    '''