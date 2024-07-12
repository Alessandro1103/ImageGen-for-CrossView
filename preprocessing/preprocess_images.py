import pandas as pd
import cv2
import os
import numpy as np

def preprocess_image(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.concatenate((image_resized, edges_colored), axis=2)
    return combined

def preprocess_and_save_images(csv_file, images_base_dir, output_dirs, column_names, target_size=(512, 512)):
    os.makedirs(output_dirs, exist_ok=True)  # Assicurati che la directory di output esista
    df = pd.read_csv(csv_file, names=column_names)
    
    for index, row in df.iterrows():
        for col, img_path in zip(column_names, row):
            full_image_path = os.path.join(images_base_dir, img_path)
            if os.path.exists(full_image_path):
                processed_image = preprocess_image(full_image_path, target_size)
                output_path = os.path.join(output_dirs, os.path.basename(img_path))
                cv2.imwrite(output_path, processed_image)
            else:
                print(f"Image not found: {full_image_path}")

if __name__ == "__main__":
    base_dir = '../CVUSA_subset'
    csv_dir = os.path.join(base_dir, 'csv')  # Aggiungi una sottocartella per i file CSV se necessario
    
    output_base = '../processed_images'  # Cartella base per le immagini processate
    output_dirs_train = os.path.join(output_base, 'train')
    output_dirs_val = os.path.join(output_base, 'val')

    preprocess_and_save_images(
        csv_file=os.path.join(csv_dir, 'train-19zl.csv'),
        images_base_dir=base_dir,
        output_dirs=output_dirs_train,
        column_names=['bingmap', 'streetview_1', 'streetview_2'],
        target_size=(512, 512)
    )
    preprocess_and_save_images(
        csv_file=os.path.join(csv_dir, 'val-19zl.csv'),
        images_base_dir=base_dir,
        output_dirs=output_dirs_val,
        column_names=['bingmap', 'streetview_1', 'streetview_2'],
        target_size=(512, 512)
    )

    print("Data preparation completed.")
