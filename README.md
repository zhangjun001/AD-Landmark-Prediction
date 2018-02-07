# AD-Landmark-Prediction

Keras implementation for predefined AD landmark prediction.



The code was written by Jun Zhang and Mingxia Liu, Department of Radiology at UNC. Jun Zhang is currently working at Duke University

Applications

Predict 50 AD landmarks



Prerequisites

Linux python 2.7

Keras version 0.2.0_4

NVIDIA GPU + CUDA CuDNN (CPU mode, untested) Cuda version 8.0.61
                        
Getting Started

Installation

Install Keras and dependencies 

Install SimpleITK with pip install SimpleITK

Install numpy, scipy, and matplotlib with pip install numpy scipy matplotlib

Download the pretrained model from https://duke.box.com/s/bme1f2tnk4vefomlbxyal5y17c6ah2al

Copy the model to the folder of /Model

cd to current folder and 

Apply our Pre-trained Model with GPU



python main.py --path Img.nii.gz

Note we use the Keras backend as follows
{
    "image_data_format": "channels_first",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

Citation

If you use this code for your research, please cite our paper:

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

@article{liu2018landmark,
  title={Landmark-based deep multi-instance learning for brain disease diagnosis},
  author={Liu, Mingxia and Zhang, Jun and Adeli, Ehsan and Shen, Dinggang},
  journal={Medical image analysis},
  volume={43},
  pages={157--168},
  year={2018},
  publisher={Elsevier}
}

