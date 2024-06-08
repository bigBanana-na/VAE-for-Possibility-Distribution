# VAEPd - Variational Autoencoder for Probability Distributions

This repository contains an implementation of a Variational Autoencoder (VAE) by Pytorch designed to learn and generate data from a given probability distribution. The model uses a variational approach to encode input data into a latent representation and then decode it back to generate new data samples.

## Overview

The goal of this VAE model is to learn a mapping from an input distribution \( q \) to a target distribution \( p \). By training the model with samples from \( q \), we aim to generate samples that are similar to those from \( p \). This can be useful in various applications such as data generation, domain adaptation, and data smoothing.

## Model Architecture

The model consists of two main components:

1. **Encoder**: Maps input data to a latent space by outputting the mean and log-variance of a Gaussian distribution.
2. **Decoder**: Generates new data samples from the latent space.

### Encoder

The encoder network consists of:
- An input layer
- A hidden layer with ReLU activation
- An output layer that generates the mean and log-variance of the latent variables

### Decoder

The decoder network consists of:
- An input layer that takes the latent variables
- A hidden layer with ReLU activation
- An output layer with Softmax activation to produce the final probability distribution

## Loss Function

The loss function used to train the VAE consists of two parts:
- **Reconstruction Loss**: Measures the difference between the input data and the reconstructed data. In this implementation, we use the cross-entropy loss.
- **KL Divergence Loss**: Measures the difference between the learned latent distribution and the prior distribution (assumed to be a standard normal distribution).
