# SRGAN
A project dedicated to the study of the architecture of a SRGAN. 
A detailed description of the model's intricacies can be found in the article [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network](https://arxiv.org/pdf/1609.04802#page=10&zoom=100,66,644) <br>
Notebook : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VQPwycLFqisijA5ynduMg3s9-EghLvTk?usp=sharing)

## Project goal
*   Study of the above article.
*   Writing the architecture of the model and preparing for training.
*   Train Super-Resolution GAN on [div2k](https://www.kaggle.com/datasets/sharansmenon/div2k) and [flickr2k](https://www.kaggle.com/datasets/hliang001/flickr2k) datasets.
*   Analyze the accuracy of the improved images.
## Components
*   **`dsdl.py`**: Loads an ImageDataset and ImageDataloader. It is nessasary in order to properly prepare the training data.
*   **`discriminator.py`**: Loads a Discriminator class.
*   **`generator.py`**: Loads a Genertor class.
*   **`training.py`**: Imports training functhion that starts the training process with nessasary 
*   **`test.py`**: Demonstrates how to use the trained SRGAN to improve quality of images.

## Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision Pillow tqdm # Add other dependencies if needed

## Architecture and Loss of models
<br> ![](architecture.png) <br>
The architecture was taken from the article. The difference from the article is a simplified generator loss function(vgg19->vgg16).
```
class GeneratorLoss(nn.Module):
  def __init__(self):
    super(GeneratorLoss, self).__init__()
    vgg = vgg16(weights='IMAGENET1K_V1')
    loss_network = nn.Sequential(*list(vgg.features)[:29]).eval()
    for param in loss_network.parameters():
        param.requires_grad = False

    #Perceptual Loss
    self.loss_network = loss_network
    self.per_loss = nn.MSELoss()
    #Adversarial loss
    self.adv_loss = nn.BCELoss()
    #L1
    self.l1_loss = nn.L1Loss()

  def forward(self, pred_image, target_image, pred, target):
    loss1 = self.per_loss(self.loss_network(pred_image), self.loss_network(target_image))
    loss2 = self.adv_loss(pred, target)
    loss3 = self.l1_loss(pred_image, target_image)
    return 0.006 * loss1 + 0.001 * loss2 + 0.02 * loss3
```
<br> Discriminator loss does not differ <br>
```
class DiscriminatorLoss(nn.Module):
  def __init__(self):
    super(DiscriminatorLoss, self).__init__()
    self.loss = nn.BCELoss()
  def forward(self, pred, target):
    return 10 * self.loss(pred, target)
```
Тhe training took place in two stages.
- Training generator with only MSE loss or L1 loss.
- Training generator with GeneratorLoss(presented above).

