# classify-audio

"Small-data" project on audio classification.

This is a study on the composition of Convolutional Neural Networks (CNNs) and how different activation functions within each convolution block affects cross-entropy and accuracy performance of a given classifier model. The dataset, data pre-processsing/loading methods, and structure of the initial ReLU CNN were created with the guidance of Valerio Velardo's "[PyTorch for Audio + Music Processing](https://github.com/musikalkemist/pytorchforaudio)" YouTube series. Networks are trained on the public [UrbanSound8K](https://urbansounddataset.weebly.com) dataset, an archive of 8732 labeled sound excerpts of up to 4 seconds of 10 different urban sounds:

1.  Air conditioner

2.  Car horn

3.  Children playing

4.  Dog bark

5.  Drilling

6.  Engine idling

7.  Gunshot

8.  Jackhammer

9.  Siren

10. Street music

J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.\
