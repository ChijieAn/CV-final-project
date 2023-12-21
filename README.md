# CV-final-project
This is the repository for Computer Vision final course project. We trained classifiers to guide DDPM image generation. 

# The CV Final Project Notebook folder
This folder consists of three notebooks I use to train the models, genertate images and evaluate the results

In the train_classifier file, we implemented two approaches of training classification models. The first approach it to train a classifier on the whole cifar-10 dataset, and another is to train a sequence of classifiers on the images with different timesteps of noise added onto it. 

In the Classifier analysis file, we analysed the confusion matrix and accuracy of hte classifier we have changed

In the cifar10_DDPM_ipynb file, we used the classifiers we have trained to guide the image generation, and the result turned out that almost all the image will be classified to the class we desire if we set a proper classifier scale, which controls the generation process. 

# The Trained Classifiers folder
This folder consists of the classifiers I trained on cifar-10 dataset which will be used to guide the generation of the model

# DDPM_image_generation folder
This folder consists of four python files.

In Gaussian Diffusion.py, I defined a Gaussion Diffusion class which will be used to similate the whole structure of DDPM model

In Unet Model.py, I defined a Unet model

In train Unet.py, I defined the function for training the Unet model

In image generation.py, I launched experiments attempts in image generation using DDPM

There's also a file about the parameters of Unet model which is finetuned 200 epochs on Cifar-10 dataset, but the file size exceeds the size limit of github so I can't push it to github 

# Training of Classifier folder
This folder consists of three python files

In get_noisy_dataset.py, I defined some functions to help generate the dataset which is used to train the desired classifier model

In train_function.py, I defined the training and evaluation functions

In train.py, I launched the training process for desired classifiers.

