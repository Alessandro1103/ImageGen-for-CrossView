import pandas as pd
import cv2
import os
import numpy as np

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Carica l'immagine, la ridimensiona e applica il rilevamento dei bordi.
    """
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    combined = np.dstack((image_resized, edges))
    return combined

def preprocess_and_save_images(csv_file, source_dir, output_dir, image_column, target_size=(512, 512)):
    """
    Legge le immagini dal CSV, le preelabora e salva i risultati.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    for image_name in df[image_column]:
        image_path = os.path.join(source_dir, image_name)
        if os.path.exists(image_path):
            preprocessed_image = preprocess_image(image_path, target_size)
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, preprocessed_image)
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    # Percorsi
    base_dir = '/home/mbrapa/University/CV_project-1/CVUSA_subset'
    
    preprocess_and_save_images(
        csv_file=os.path.join(base_dir, 'train-19zl.csv'),
        source_dir=os.path.join(base_dir, 'bingmap'),
        output_dir=os.path.join(base_dir, 'preprocessed_bingmap_train'),
        image_column='bingmap'
    )
    preprocess_and_save_images(
        csv_file=os.path.join(base_dir, 'val-19zl.csv'),
        source_dir=os.path.join(base_dir, 'bingmap'),
        output_dir=os.path.join(base_dir, 'preprocessed_bingmap_val'),
        image_column='bingmap'
    )
    preprocess_and_save_images(
        csv_file=os.path.join(base_dir, 'train-19zl.csv'),
        source_dir=os.path.join(base_dir, 'streetview'),
        output_dir=os.path.join(base_dir, 'preprocessed_streetview_train'),
        image_column='streetview'
    )
    preprocess_and_save_images(
        csv_file=os.path.join(base_dir, 'val-19zl.csv'),
        source_dir=os.path.join(base_dir, 'streetview'),
        output_dir=os.path.join(base_dir, 'preprocessed_streetview_val'),
        image_column='streetview'
    )

    print("Data preparation completed.")
