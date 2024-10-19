## AlexNet-Transfer-Learning-on-MNIST

## Overview
This project implements a transfer learning approach using a pre-trained **AlexNet** model on the **Fashion-MNIST** dataset. By fine-tuning the final layer of the AlexNet model, we demonstrate how pre-trained models can be used effectively for new tasks with minimal modifications.

## Dataset
- **Fashion-MNIST** dataset consists of 60,000 training images and 10,000 test images, with 10 classes of clothing items such as T-shirts, coats, dresses, etc.
- Each image is a 28x28 grayscale image, resized to 224x224 to match the input requirements of AlexNet.

## Model
- The **AlexNet** model, pre-trained on the ImageNet dataset, is used as the base model.
- All layers except the final fully connected layer are frozen, and the final layer is replaced to predict 10 clothing categories from the Fashion-MNIST dataset.

## Training
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: CrossEntropyLoss.
- **Epochs**: The model is trained for 10 epochs.
- **Batch Size**: 32

## Results
- The fine-tuned model achieves over **90% accuracy** on the test set.
- This demonstrates the power of transfer learning by leveraging pre-trained models for faster training and higher accuracy on small datasets.
