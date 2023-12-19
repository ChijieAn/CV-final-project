# CV-final-project
This is the repository for Computer Vision final course project. We trained classifiers to guide DDPM image generation. 

In the train_classifier file, we implemented two approaches of training classification models. The first approach it to train a classifier on the whole cifar-10 dataset, and another is to train a sequence of classifiers on the images with different timesteps of noise added onto it. 

In the Classifier analysis file, we analysed the confusion matrix and accuracy of hte classifier we have changed

In the cifar10_DDPM_ipynb file, we used the classifiers we have trained to guide the image generation, and the result turned out that almost all the image will be classified to the class we desire if we set a proper classifier scale, which controls the generation process. 
