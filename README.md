# Building a Facial Recognition API

![img](https://miro.medium.com/max/1400/1*DKSQVZdEa2GEv2ksxWViTg.gif)

### Our project
In this project we are going to build a CNN that will help us compare if to pictures are of the same person. 📸

We will use the dataset of [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/#download) enriched with some pictures of ourselves.🤩

This model is a binary prediction model.  
For a better understanding we will set two terminology words:
- `Anchor`:  The picture to compare to.  
- `Input`: The picture that is being compared.

### Baseline Model
Our baseline model is a `Gaussian Naive Bayes Classifier` with an accuracy of `51%` (random guess).

The anchor and input are flattened and concatenated together to form a one 375,001 size vector (1 for the label) and sent into the GNB model

We do not expect to get good results from this model. Getting 51% accuracy assures us that we are on the right path.

### Twin CNN:
- The model will get the two images after preprocessing:
  - Anchor (224, 224, 3)
  - Input (224, 224, 3)
- The two images will go through the VGG16 embedding stage. While the model trains it trains on the same embedding for both images.
- Out of the embedding we get two vectors that enter an L1 distance layer. Here the distance between the two images is measured. 
- One last dense layer with a sigmoid activation to get us a binary classification.

![img](https://ashvijay.github.io/assets/img/STN.jpg)

## Project stages
**Let’s start! 🚀**

### 1.	Downloading the LFW dataset and enriching it with pictures of ourselves
* Downloading LFW and adding pictures of ourselves.
* Extracting pictures of people that has more than one picture in their folder to a folder named data (including the new pictures)

### 2. Dataset building
getting image from the data folder as anchor.

For positive:  
Taking another image from the directory of the person that in the anchor

For the negative:
Taking a random image of a different person from the data folder

(At the training we will use a Keras custom image generator to load the data)

### 3. Preprocessing
Resizing and rescaling the data.

### 4. Training the model
The model was trained using ADAM optimizer in three stages, each stage took 50 epochs, with the exception of early stopping if necessary:
- 1st stage: lr = 1e-4
- 2nd stage: lr = 1e-5
- 3rd stage: lr = 1e-6  
  (lr: learning rate)

That was to ensure that the model will have a low loss as possible.

### 4. Testing the model
#### Setting up voters:
Taking a set of voter images to place as the anchors to the input image.
#### Comparing input to voters:
An image will be taken (the input) and then be compared to a set of voter images. The average of the outcomes will be the confidence of the model if the person is the same as the voters or not

### Prerequirements:
-	conda \ pip Install **requirements.txt**
-	**GPU required for training**. This project trains a CNN, so either work on local GPU or use a cloud GPU.

#### Libraries:
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
- [glob](https://docs.python.org/3/library/glob.html)
- [collections](https://docs.python.org/3/library/collections.html)
- [random](https://docs.python.org/3/library/random.html)

### Contents of the repository:
- `baseline_model.ipynb`: where the baseline model is created
- `baseline_model.py`: where the baseline model functions are created
- `config.py`: with the constants
- `get_samples.ipnyb`: where we create the dataset with pictures of ourselves
- `preprocessing.py`: where the preprocessing functions take place
- `requirements.txt`
- `creating_DF.ipynb`: where all the images from the folders were reorganized so that we have a balanced dataset.
- `creating_DF.py`: where the functions needed for creating the DF are stored.
- `img_generator.py`: Custom image generator
- `train_NN.py`: Training loop for CNN
- `voters.py`: Function that takes pictures to set as voters
- `model_test_real_time.py`: A test function for trained model

### References
[Youtube Video](https://www.youtube.com/watch?v=LKispFFQ5GU)  
[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

### Authors 
> Authors: Noam Cohen, Sahar Garber and Julieta Staryfurman

# Important For GitHub Cloning
This repo has some large files. To get them we use git lfs (large file storage).  
To use git lfs go to [here](https://git-lfs.github.com/) and download git lfs to your machine. After that enter in command line in the git repo:  
`$ git lfs install`  
Then, as you do with normal git, pull the repo from GitHub and the large files will be tracked by git lfs and downloaded to your machine.

