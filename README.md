# Building a Facial Recognition API

![img](https://miro.medium.com/max/1400/1*DKSQVZdEa2GEv2ksxWViTg.gif)

### Our project
In this project we are going to build a NN that will help us compare if to pictures are of the same person. 📸

We will use the dataset of [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/#download) enriched with some pictures of ourselves to make sure that we have enough amount of samples of the same person.🤩 This is what the dataset looks like in some examples:

(images from EDA)

In order to do our predictions we are going to base our model in similarities and that’s why  it’s important that we divide our images in:

`Anchor dataset`:  actual image. The picture to compare to.  

`Positive verification dataset`: image of the same person. It is going to be an image of the same person as in the anchor.  (1 - True)

`Negative verification dataset`: image of a different person. It is going to be an image of the LFW dataset.  (0 - False)

We’re going to compare the anchor image with the input image and it will compare if both images are the same person. 

Before we start, we will separate our data in this 3 folders: `positive, negative and anchor`. 

**Let’s start! 🚀**

### 1.	Build the dataset with images of ourselves: we are going to use Open CV which allows the access to our camera. We need to make sure the resolution of the image is 255x255. 

`a`: takes an anchor picture
`p`: takes a positive picture
`q`: closes the camera

### 2.	Preprocessing
In order to preprocess we need to resize and scale the data. 


### Prerequirements:
-	Install requirements.txt
-	Work with Google Colab using their GPU

### Libraries:
- [cv2](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [os](https://docs.python.org/3/library/os.html)
- [numpy](https://numpy.org/doc/)
- [tensorflow]( https://www.tensorflow.org/)
- [matplotlib]( https://matplotlib.org/)


### Contents of the repository:
-	Taking pictures of ourselves
-	EDA + Preprocessing
-	Baseline model


### References
[Youtube Video](https://www.youtube.com/watch?v=LKispFFQ5GU)
[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

### Authors 
> Authors: Noam Cohen, Sahar Garber and Julieta Staryfurman