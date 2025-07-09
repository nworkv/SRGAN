# SRGAN
A project dedicated to the study of the architecture of a SRGAN. 
A detailed description of the model's intricacies can be found in the article [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network](https://arxiv.org/pdf/1609.04802#page=10&zoom=100,66,644)
## Setup 
For all the methods described in the paper, is it required to have:
- Anaconda

Specific requirements for each method are described in its section. 
To install SRGAN please run the following commands:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install git+https://github.com/nworkv/SRGAN.git
```
