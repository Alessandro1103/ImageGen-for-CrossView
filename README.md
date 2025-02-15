# Generation of Aerial Images from Ground-Level Views for Cross-View Matching

## Abstract
```
ImageGen-for-CrossView/
│── CVUSA_subset/
│   ├── bingmap/
│   ├── streetview/
│
│── FeatureExtractor/
│   ├── VGG.py
│
│── JoinFeatureLearning/
│   ├── JFL.py
│   ├── main.py
│
│── models/
│
│── XFork/
│   ├── discriminator.py
│   ├── generator.py
│   ├── main.py
│
├── blocks.py
├── dataset.py
├── edge_Concatenate.py
├── eval.py
├── meansOfImages.py
│
│── Presentation/
│   ├── Computer_Vision_Presentation/
│
│── Sources/
│   ├── 1904.11045v2.pdf
│   ├── Regmi_Cross-View_Image_Sy...
│
│── .gitignore
│── README.md
```
## Description
This project aims to replicate "Bridging the Domain Gap for Ground-to-Aerial Image Matching" by Krishna Regmi and Mubarak Shah. The chosen structure to be replicated focuses on the GAN and Joint Feature Learning. The interest in this paper arises from the usefulness of this research in improving image representation for applications such as localization, navigation, and mapping.

## Dataset
In this project, due to limited resources, the CVUSA dataset has been reduced. The original dataset can be found here: [Complete Dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/) with an appropriate request. Alternatively, a reduced version can be found here: [Reduced Dataset](https://pro1944191.github.io/SemanticAlignNet/).

## How to Run
The paper states that the GAN is pretrained, but in this project, a codebase has been developed to replicate the training process. 

1. First, the GAN needs to be trained using the appropriate main script located in the XFork folder:
   ```
   Code/XFork/main.py
   ```
2. Once the GAN is trained, the next step is to evaluate the results using the Joint Feature Learning algorithm. The main script for this step is found in the following folder:
   ```
   Code/JoinFeatureLearning/main.py
   ```
Actually, I have trained the models via [Kaggle](https://www.kaggle.com/) and the results can be found here:
- [Gan Train](https://www.kaggle.com/code/alessandro1103/primo-train-gan)
- [JointFeatureLearning Train](https://www.kaggle.com/code/alessandro1103/notebookdc51fdc810)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Alessandro1103/ImageGen-for-CrossView.git
   ```
2. Install the required dependencies:
   ```
   pip install -r <Folder>/ImageGen-for-CrossView/requirements.txt
   ```

## Author
Alessandro De Luca

