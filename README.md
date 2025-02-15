# Generation of Aerial Images from Ground-Level Views for Cross-View Matching

## Abstract
```
ImageGen-for-CrossView/
│── CVUSA_subset/
│   ├── bingmap/
│   ├── streetview/
│
│── FeatureExtractor/
│   ├── ├── VGG.py
│
│── JoinFeatureLearning/
│   ├── JFL.py
│   ├── main.py
│
│── models/
│
├── XFork/
│   ├── discriminator.py
│   ├── generator.py
│   ├── main.py
├
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

## Dataset
In my project, due to limited resources, I have reduced the CVUSA dataset. The original dataset can be found here: {https://mvrl.cse.wustl.edu/datasets/cvusa/}(Complete Dataset) with an appropriate request. Otherwise a reduced version can be found {https://pro1944191.github.io/SemanticAlignNet/}(Reduced Dataset)

## How to run
The paper said that the GAN is pretrained, but here there is a code designed to replicate the training. So first of all we need to run the GAN, and there is the appropriate main in the XFork folder, ```Code/XFork/main.py```:
Once the GAN is trained the next step would be to evaluate the results using the JointFeatureLearning algorithm. As above the main is in the appropriate folder, ```Code/JointFeatureLearning/main.py```. 

## Installation
1. Clone
```
https://github.com/Alessandro1103/ImageGen-for-CrossView.git
```
2. Intall requirements
```
pip install -r <Folder>/ImageGen-for-CrossView/requirements.txt
```
## Author
Alessandro De Luca
