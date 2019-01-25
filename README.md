# Uncertainty Estimation with Convolutional Neural Networks and Gaussian Processes

Neural Networks have achieved state-of-the-art accuracy in many applications,
ranging from computer vision to natural language processing. However, standard
neural networks lack a principled way to measure the uncertainty in their prediction,
which is critical in sensitive applications like healthcare, trading or autonomous
vehicles. On the other hand, gaussian processes allow for a principled estimation of
uncertainty, but lack the scalability and compositionality of deep learning models.
We propose a model for image classification which combines convolutional neural
networks and gaussian processes to bring together the best of the two models. We
show that our model achieves high accuracy and scalability, while also precisely
estimating the uncertainty in its predictions. The model consistently outperforms a
convolutional neural network on a classification task where the models can reject
inputs they are not confident about. Moreover, our model is also more robust
to adversarial examples. This suggests that our model should be preferred for
sensitive applications where misclassifications are costly.

Full report can be found at https://docdro.id/9K5dVj5