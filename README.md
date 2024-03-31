# Dog Breed Detection using CNN

![Project Poster](https://github.com/rahultripathi17/Dog_Breed_Detection_ML/assets/165544212/db5b18bb-c9b2-4175-a899-6bc2cb63ad96)

## Description
Dog breed detection is the process of identifying the breed of a dog from an input image, which is a challenging computer vision task. In this project, we leverage deep learning techniques, specifically convolutional neural networks (CNN), to classify dog breeds from input images. We utilize transfer learning with a pre-trained VGG-16 model and fine-tune it for our task, achieving high accuracy on the test set.

## Report
- [Report](https://drive.google.com/file/d/15msq1hzx9o_aOHkkNWm3q4kMbUuL3MFJ/view?usp=drive_link)
- [Execution Report](https://drive.google.com/file/d/1Cc1dJSQUUp12INPewXd7C3lXG5Y-Fndt/view?usp=drive_link)
- [Poster](https://drive.google.com/file/d/1j2WXxqBIUKKuyLutPuvCjrwAtchrIfuz/view?usp=drive_link)

## Dataset and Features
We used a publicly available dataset from Kaggle consisting of over 20,000 images of dogs from 120 different breeds. The dataset was split into training and test sets, and data augmentation techniques were applied to prevent overfitting. We utilized transfer learning with the VGG16 model as a feature extractor, achieving an accuracy of 82.6% on the test set.

## Methods
We employed a CNN architecture with four layers: two convolutional layers, two max-pooling layers, and two fully connected layers. Data preprocessing involved resizing images to 64x64 pixels and converting them to RGB format. We used the Adam optimization algorithm with binary cross-entropy loss for training, along with early stopping to prevent overfitting.

## Results
Our best-performing model achieved a test accuracy of 82.1%, outperforming the baseline accuracy of random guessing. We experimented with deeper architectures and transfer learning but found that a simpler CNN architecture provided the best results. Precision and recall values were around 80%, indicating balanced performance in identifying both positive and negative cases.

## Discussion
While our results demonstrate the effectiveness of deep learning for dog breed detection, challenges remain in detecting fine-grained breeds and those with low representation in the training data. Future work could focus on improving data quality, experimenting with different architectures and hyperparameters, and exploring transfer learning for better performance.

## Future Work
- Improving data quality and quantity
- Experimenting with different network architectures and hyperparameters
- Exploring transfer learning for enhanced performance
- Applying dog breed detection to practical applications in animal welfare
