# MNIST Image Classification

This project aims to classify handwritten digits from the MNIST dataset using various techniques. The classification is initially performed using a normal neural network (1D), then extended to include convolutional layers (2D), and finally optimized using Keras Tuner.

## Dataset

The MNIST dataset is a collection of 28x28 grayscale images of handwritten digits (0-9). It contains 60,000 training images and 10,000 testing images.

## Preprocessing

Before training the models, the dataset undergoes preprocessing steps such as normalization and reshaping to prepare it for input into the neural network models.

## Model Architectures

### Normal Neural Network (1D)

The initial approach utilizes a simple feedforward neural network with densely connected layers.

### Convolutional Neural Network (2D)

To capture spatial features in the images, convolutional layers are introduced, which are particularly effective for image classification tasks.

### Keras Tuner Optimization

Keras Tuner is employed to automatically search for the optimal hyperparameters of the convolutional neural networks(such as the number of filter sizes, layers, dropout etc).

## Results

The performance of each model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Dependencies

The project is implemented in Python using the following libraries:

- TensorFlow
- Keras
- Keras Tuner
- NumPy
- Matplotlib

## Kaggle Link:
https://www.kaggle.com/code/dsingh9/cnn-mnist/notebook

## License

This project is licensed under the [MIT License](LICENSE).
