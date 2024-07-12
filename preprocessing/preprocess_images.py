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

def preprocess_and_save_images(csv_file, base_dir, output_dirs, target_size=(512, 512)):
    """
    Legge le immagini dal CSV, le preelabora e salva i risultati.
    """
    # Crea le directory di output se non esistono già
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)

    # Carica il file CSV
    df = pd.read_csv(csv_file)
    
    # Processa ogni riga del CSV
    for index, row in df.iterrows():
        for col, image_path in row.items():
            full_image_path = os.path.join(base_dir, image_path)
            if os.path.exists(full_image_path):
                preprocessed_image = preprocess_image(full_image_path, target_size)
                output_path = os.path.join(output_dirs[col], os.path.basename(image_path))
                cv2.imwrite(output_path, preprocessed_image)
            else:
                print(f"Image not found: {full_image_path}")

if __name__ == "__main__":
    # Percorsi
    base_dir = '/home/mbrapa/University/CV_project/CVUSA_subset'
    
    output_dirs_train = {
        'bingmap': os.path.join(base_dir, 'preprocessed_bingmap_train'),
        'streetview_1': os.path.join(base_dir, 'preprocessed_streetview_train'),
        'streetview_2': os.path.join(base_dir, 'preprocessed_streetview_train_2')
    }
    
    output_dirs_val = {
        'bingmap': os.path.join(base_dir, 'preprocessed_bingmap_val'),
        'streetview_1': os.path.join(base_dir, 'preprocessed_streetview_val'),
        'streetview_2': os.path.join(base_dir, 'preprocessed_streetview_val_2')
    }

    # Preprocessamento e salvataggio delle immagini
    preprocess_and_save_images(
        csv_file=os.path.join(base_dir, 'train-19zl.csv'),
        base_dir=base_dir,
        output_dirs=output_dirs_train,
        target_size=(512, 512)
    )
    preprocess_and_save_images(
        csv_file=os.path.join(base_dir, 'val-19zl.csv'),
        base_dir=base_dir,
        output_dirs=output_dirs_val,
        target_size=(512, 512)
    )

    print("Data preparation completed.")
