import torch
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image  # Importiamo PIL per garantire la compatibilità con ToTensor()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder):
        self.sat_folder = os.path.join(root_folder, "bingmap")
        self.street_folder = os.path.join(root_folder, "streetview")

        self.sat_images = sorted(os.listdir(self.sat_folder))
        self.street_images = sorted(os.listdir(self.street_folder))

        assert len(self.sat_images) == len(self.street_images), (
            f"Le due cartelle devono contenere lo stesso numero di immagini: "
            f"Bingmap = {len(self.sat_images)}, Streetview = {len(self.street_images)}"
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        sat_path = os.path.join(self.sat_folder, self.sat_images[idx])
        street_path = os.path.join(self.street_folder, self.street_images[idx])

        # Legge le immagini con OpenCV
        image_sat = cv2.imread(sat_path)
        image_street = cv2.imread(street_path)

        # Convertire in formato RGB (OpenCV usa BGR)
        image_sat = cv2.cvtColor(image_sat, cv2.COLOR_BGR2RGB)
        image_street = cv2.cvtColor(image_street, cv2.COLOR_BGR2RGB)

        # Converte le immagini in formato PIL perché ToTensor() lavora meglio con PIL
        image_sat = Image.fromarray(image_sat)
        image_street = Image.fromarray(image_street)

        # Applica la trasformazione
        image_sat = self.transform(image_sat)
        image_street = self.transform(image_street)

        assert image_sat.shape[0] == 3, f"Immagine non a 3 colori: {image_sat.shape[0]}"
        assert image_street.shape[0] == 3, f"Immagine non a 3 colori: {image_street.shape[0]}"

        resize_transform_sat = transforms.Resize((512, 512))
        if image_sat.shape[1:] != (512, 512):
            image_sat = resize_transform_sat(image_sat)

        resize_transform_street = transforms.Resize((224, 1232))
        if image_street.shape[1:] != (224, 1232):
            image_street = resize_transform_street(image_street)

        assert image_sat.shape == torch.Size([3,512,512]), f"Shape errata: {image_sat.shape}"
        assert image_street.shape == torch.Size([3,224,1232]), f"Shape errata: {image_street.shape}"

        return image_sat, image_street

    def __len__(self):
        return len(self.sat_images)



def test():
    dataset = ImageDataset("./CVUSA_subset")
    image_sat, image_street = dataset[0]

    # Le immagini PyTorch hanno shape (C, H, W) → dobbiamo convertirle in (H, W, C) per Matplotlib
    image_sat = image_sat.permute(1, 2, 0).cpu().numpy()
    image_street = image_street.permute(1, 2, 0).cpu().numpy()

    # Denormalizziamo per riportare i valori nell'intervallo [0,1]
    image_sat = (image_sat * 0.5) + 0.5
    image_street = (image_street * 0.5) + 0.5

    # Creiamo una figura con due subplot per le due immagini
    plt.figure(figsize=(10, 5))

    # Mostra l'immagine satellitare
    plt.subplot(1, 2, 1)
    plt.imshow(image_sat)
    plt.title("Satellite Image")
    plt.axis("off")

    # Mostra l'immagine street view
    plt.subplot(1, 2, 2)
    plt.imshow(image_street)
    plt.title("Street View Image")
    plt.axis("off")

    # Mostra il plot
    plt.show()

    # Split del dataset in Train (70%), Test (15%), Validation (15%)
    test_size = 0.15
    val_size = 0.15

    test_amount = int(len(dataset) * test_size)
    val_amount = int(len(dataset) * val_size)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
        len(dataset) - (test_amount + val_amount),
        test_amount,
        val_amount
    ])

    print(f"Train set: {len(train_set)}")
    print(f"Validation set: {len(val_set)}")
    print(f"Test set: {len(test_set)}")


if __name__ == "__main__":
    test()
