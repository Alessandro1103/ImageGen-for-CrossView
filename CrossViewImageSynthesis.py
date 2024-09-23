import os
import torch
import torchvision as vision
from tqdm import tqdm 


def calculate_mean_std(image_tensor):
    """
    Calcola la media e la deviazione standard per ciascun canale di un'immagine.
    
    :param image_tensor: Il tensore dell'immagine di dimensioni [C, H, W]
    :return: Una tupla con la media e la deviazione standard per ogni canale.
    """

    if image_tensor.max() > 1.0:
        image_tensor = image_tensor.float() / 255.0
    
    mean = image_tensor.mean(dim=[1, 2])
    std = image_tensor.std(dim=[1, 2])
    
    return mean, std



def calculate_folder_mean_std(folder_path):
    """
    Calcola la media e la deviazione standard per ogni cartella.
    
    :param image_paths: Percorso per la cartella.
    :return: Media e deviazione standard della cartella.
    """
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    num_images = 0
    
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        # Verifica se è un file immagine (puoi personalizzare i formati accettati)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_tensor = vision.io.read_image(file_path).float() / 255.0
            
            # Calcola la media e std per l'immagine
            mean, std = calculate_mean_std(img_tensor)
            
            # Somma media e std
            mean_sum += mean
            std_sum += std
            num_images += 1
    
    # Media complessiva del dataset
    dataset_mean = mean_sum / num_images
    dataset_std = std_sum / num_images
    
    return dataset_mean, dataset_std


# folder = 'path to folder' es: folder = 'streetview'
# streetview_mean, streetview_std = calculate_folder_mean_std(folder)