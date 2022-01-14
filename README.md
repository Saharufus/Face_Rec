Building a Facial Recognition API

In this project we are going to build a NN that will help us identify different people in pictures. 📸

We will use the dataset [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/#download) enriched with some pictures of ourselves to make it more fun.🤩 This is what the dataset looks like in some examples:

(images from EDA)

In order to do our predictions we are going to base our model in similarities and that’s why  it’s important that we divide our images in:

Anchor dataset:  actual image. It is going to be an image of ourselves. 
Positive verification dataset: image of the same person. It is going to be an image of ourselves.  (1 - True)
Negative verification dataset: image of a different person. It is going to be an image of the LFW dataset.  (0 - False)

We’re going to compare the anchor image with the positive and negative examples to verify if they are (or not) the same person. 

Before we start, we will separate our data in this 3 folders: positive, negative and anchor. 

Let’s start! 🚀

1.	Enrich the dataset with images of ourselves: we are going to use Open CV which allows the access to our camera. We need to make sure the resolution of the image is 255x255. 

a: takes an anchor picture
p: takes a positive picture
q: closes the camera

2.	Preprocessing
In order to preprocess we need to resize and scale the data. 


Prerequierements:
-	Install requierements.txt
-	Work with Google Colab using their GPU

Libraries:
[cv2](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
[os](https://docs.python.org/3/library/os.html)
[random](https://docs.python.org/3/library/random.html)
[numpy](https://numpy.org/doc/)
[tensorflow]( https://www.tensorflow.org/)
[matplotlib]( https://matplotlib.org/)


Contents of the repository:
-	Enriching dataset
-	EDA + Preprocessing
-	Baseline model


References
https://www.youtube.com/watch?v=LKispFFQ5GU

