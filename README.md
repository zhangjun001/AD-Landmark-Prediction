# AD-Landmark-Prediction

Keras implementation for the prediction of predefined AD landmarks.


The code was written by Jun Zhang and Mingxia Liu, Department of Radiology at UNC. Jun Zhang is currently working at Duke University

Applications

Predict 50 AD landmarks
We used group comparison to define more than 1000 landmarks that are related to the Alzheimer's disease[1]. In our following work[2], we noticed that top 20 to 50 landmarks may be enough for AD classification by intergating deep neural networks. Therefore, we provide the landmark detection code to predict 50 landmarks for new images. 

Note that, you must perform the linear alignment for your image (to Img.nii.gz), and then run our code for landmark detection.

Prerequisites

Linux python 2.7

Keras version 0.2.0_4

NVIDIA GPU + CUDA CuDNN (CPU mode, untested) Cuda version 8.0.61
                        
Getting Started

Installation

Install Keras and dependencies 

Install SimpleITK with pip install SimpleITK

Install numpywith pip install numpy 

Download the pretrained model from https://duke.box.com/s/bme1f2tnk4vefomlbxyal5y17c6ah2al

Copy the model to the folder of /Model

cd to current folder and 

Apply our Pre-trained Model with GPU



python main.py --input Img.nii.gz

Note we use the Keras backend as follows
{
    "image_data_format": "channels_first",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

Citation

If you use this code for your research, please cite our paper:

[1] 

@article{zhang2016detecting,
  title={Detecting anatomical landmarks for fast Alzheimer’s disease diagnosis},
  author={Zhang, Jun and Gao, Yue and Gao, Yaozong and Munsell, Brent C and Shen, Dinggang},
  journal={IEEE transactions on medical imaging},
  volume={35},
  number={12},
  pages={2524--2533},
  year={2016},
  publisher={IEEE}
}

[2] 

@article{liu2018landmark,
  title={Landmark-based deep multi-instance learning for brain disease diagnosis},
  author={Liu, Mingxia and Zhang, Jun and Adeli, Ehsan and Shen, Dinggang},
  journal={Medical image analysis},
  volume={43},
  pages={157--168},
  year={2018},
  publisher={Elsevier}
}

