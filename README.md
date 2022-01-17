# Building a Facial Recognition API

![img](https://miro.medium.com/max/1400/1*DKSQVZdEa2GEv2ksxWViTg.gif)

### Our project
In this project we are going to build a NN that will help us compare if to pictures are of the same person. 📸

We will use the dataset of [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/#download) enriched with some pictures of ourselves to make sure that we have enough amount of samples of the same person.🤩 This is what the dataset looks like in some examples:

(images from EDA)

In order to do our predictions we are going to base our model in similarities and that’s why  it’s important that we divide our images in (`triplet loss`):

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

### 2.	Preprocessing and EDA
In order to preprocess we need to resize and scale the data. 

### 3. Baseline Model
Our baseline model is a `Gaussian Naive Bayes Classifier` with an accuracy of `95%`.

In this step, the dataframe consists on 200 rows with pictures of ourselves. For each picture we have 11025 columns that correspond to the pixels of the anchor image and another 11025 that are the pixels of the picture that we are comparing to the anchor (in same cases positive and in other cases negative). The label is 1 if the images are the same or 0 if they are not.

### Prerequirements:
-	Install requirements.txt
-	Work with Google Colab using their GPU

### Libraries:
- [cv2](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [os](https://docs.python.org/3/library/os.html)
- [numpy](https://numpy.org/doc/)
- [tensorflow]( https://www.tensorflow.org/)
- [matplotlib]( https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/docs/)
- [numpy](https://numpy.org/doc/)
- [sklearn](https://scikit-learn.org/stable/)
- [warnings](https://docs.python.org/3/library/warnings.html)
- [seaborn](https://seaborn.pydata.org/)


### Contents of the repository:
- `baseline_model.ipynb`: where the baseline model is created
- `baseline_model.py`: where the baseline model functions are created
- `conf.py`: with the constants
- `get_samples.ipnyb`: where we create the dataset with pictures of ourselves
- `preprocessing.py`: where the preprocessing functions take place
- `data`: a folder with the 3 folders: `anchor`, `positive` and `negative`.
- `requirements.txt`


### References
[Youtube Video](https://www.youtube.com/watch?v=LKispFFQ5GU)
[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

### Authors 
> Authors: Noam Cohen, Sahar Garber and Julieta Staryfurman