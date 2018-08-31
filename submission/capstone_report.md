# Machine Learning Engineer Nanodegree

## Capstone Project
Dominik Söllinger  
August 31st, 2018

## Adaptive foreground-background segmentation using GMMs
### 1. Project Overview

Background subtraction is a common pre-processing step in many computer vision related tasks. Many applications [1][2] only require information about changes in a scene since such regions typically contain the objects we are interested in. For instance, cars that drive on a highway or pedestrians the walk on a sidewalk. Background segmentation allows us to detect such objects by a pixel-wise segmentation of frames into foreground and background regions.

Deep learning enthusiast might now suggest the use of novel neural networks like UNET [3] to solve such segmentation tasks. However, there are well-studied algorithms like the one proposed by Stauffer and Grimson [4] that provide an unsupervised learning based solution based on Gaussian mixture models (GMMs) to the segmentation problem. The proposed algorithm is also special due to its ability to actively adapt and learn from previously unseen frames. It doesn't require a prior training phase or training data. This advocates its usage for foreground-background segmentation tasks.

### 2. Problem Statement

The goal of this project is to develop a solution for adaptive foreground-background video segmentation and to assess its performance. The solution is based on the algorithm introduced by Stauffer and Grimson [4].
The model's performance gets evaluated on the LASIESTA [5] dataset. A good model should finally be able to accurately segment any video into a foreground or background regions after "seeing" a few frames and actively adapt itself to scenery changes. Ideally, the model should be able to perform this segmentation task in real-time.

### 3. LASIESTA dataset

As already mentioned the LASIESTA [5] dataset is used to train and test the implemented model. The dataset comprises of 28 different scenes captured in different indoor and outdoor environments. However, in course of this project we focus on indoor scenes only. Scenes cover a broad spectrum of image distortions we may encounter in real-world like illumination changes, occlusion or shadows.  

In each scenario we are given a set of frames corresponding to a video. Additionally, labelled images (ground truth) are available for each frame.

Colors of these labelled frames represent different types of objects:

- Black pixels (0,0,0): Background.
- Red pixels (255,0,0): Moving object with label 1 along the sequence.
- Green pixels (0,255,0): Moving object with label 2 along the sequence.
- Yellow pixels (255,255,0): Moving object with label 3 along the sequence.
- White pixels (255,255,255): Moving objects remaining static.
- Gray pixels (128,128,128): Uncertainty pixels.

In course of this project we will focus on the following eight different indoor scenes:

| Scene   | Description |
| --------|-------------|
| I\_SI\_01 | Sequences not containing camouflage, occlusions, illumination changes, modified background, camera motion, or bootstrapping.|
| I\_CA\_01 | Sequences with moving objects remaining temporally static on background regions with similar color.|
| I\_OC\_01 | Sequences containing totally or partially occluded moving objects. |
| I\_IL\_01 | Sequences with global illumination changes. |
| I\_MB\_01 | Sequences showing situations in which background elements are subtracted or where some objects are abandoned. |
| I\_BS\_01 | Sequences containing moving objects from the first frame. |
| I\_MC\_01 | Sequences recorded with non-completely static cameras (handy cameras or pan/tilt motion). |
| I\_SM\_01 | Set of sequences simulating different types and intensities of camera motion.|


<img src="images/sample_orig_img_1.bmp" height="130px" /> <img src="images/sample_seg_img_1.png" height="130px" /> <img src="images/sample_orig_img_2.bmp" height="130px" /> <img src="images/sample_seg_img_2.png" height="130px" />


### 4. GMM for segmentation tasks

As we focus on solving this problem by means of a particular class of algorithms, namely, Gaussian Mixture models, it's important to have a general understanding of how these algorithms work and how they can be applied to solve video segmentation tasks. This understanding will help us to understand what kind of information we have to look at when analyzing the dataset.

The idea is to fit a Gaussian mixture model to a time series of image pixels. In other words, we are given a video and take a certain number of frames out of this video. Next, we consider what's called a "pixel process". The "pixel process" is a time series of pixel values, for example, scalars for gray values or vectors for color images. 

At any time $t$ we know the history (intensity values) of a certain pixel:

<center>$\{X_1,...,X_t\} = \{\hspace{0.2cm} I(x,y,i): 1 \leq i \leq t \hspace{0.2cm} \} \hspace{1cm}$ where $I$ is the image sequence</center>

We can now use this time series to estimate the probability of seeing a certain pixel value. Formally, this means that we can estimate the probability of a certain pixel value by fitting a GMM on the recent history of pixel values.

The probability of observing the current pixel value is 

<center>$P(x) = \sum_{I=1}^K w_{i,t} \cdot \eta(X_t,\mu_{i,t},\Sigma_{i,t})$</center>

where K is the number of distributions, $w_{i,t}$ is an estimate of the weight (what portion of the data is accounted for by this Gaussian) of the i-th Gaussian in the mixture at time $t$, $\mu_{i,t}$ is the mean value of the i-th Gaussian in the mixture at time $t$, $\Sigma_{i,t}$ is the covariance
matrix of the i-th Gaussian in the mixture at time $t$, and where $\eta$ is a Gaussian probability density function.

<center> $\eta(X_t,\mu_{i,t},\Sigma_{i,t}) = \frac{1}{ (2\pi)^{\frac{n}{2}} |\Sigma|^{\frac{1}{2}}}\cdot e^{-\frac{1}{2} (X_t-\mu_t)^T \Sigma^{-1} (X_t-\mu_t)}$ </center>

**Background Model estimation**

The distribution of recently observed values of each pixel in the scene is now characterized by a mixture of Gaussians. This mixture can now be used to estimate the probability that a certain pixel value belongs to a background or foreground region. The idea is similar to an approach called Bog-of-words (BoW) classification. Some Gaussians are more likely to represent a background region than others. If we know which Gaussians represents background objects, we can assign new pixel values to either a background or foreground region by calculating its proximity to background Gaussians.

To understand this, consider the accumulation of supporting evidence and relatively low
variance for "background" distributions when a static, persistent object is visible. In contrast, when a new object occludes the background object, it will not, in general, match one of the existing distributions which will result in either the creation of a new distribution
or an increase in variance of an existing distribution. Also, the variance of the moving object
is expected to remain larger than a background pixel until the moving object stops.


### 5. Analysis

#### 5.1. Data Exploration

We already discussed how GMMs can be used to segment video frames. In Layman's term we could say that our expectation is to find groups in a series of pixel values. Each group comprises of similar pixel values. If our assumption is correct, background / foreground pixel values tend to be in different categories. Therefore, let's see if such a behavior is evident in our dataset.

##### 5.1.1. Overall distribution per category

First, we look the overall distribution of pixel values by category (foreground/background) across the different scenes of the dataset. For instance, we simply take the values of all background pixels and compute statistical descriptors like mean and standard deviation.

The following tables show the result for three out of the eight scenarios we analyzes. Results for the remaining scenarios can be found in the project's repository [9].

|     I\_SI\_01          |  TOTAL (RED)  | FG (RED)  | BG (RED) | TOTAL (GREEN)  | FG (GREEN)  | BG (GREEN) | TOTAL (BLUE)  | FG (BLUE)  | BG (BLUE) |
|------------------------|---------------|-----------|----------|----------------|-------------|------------|---------------|------------|-----------|
| Mean                   | 102.29        | 86.18     | 103.07   |  100.47        | 83.97       | 101.26     | 93.73         | 75.63      | 94.58     |
| Std                    |  65.38        | 60.19     | 65.61    |  63.31         | 63.57       | 57.03      | 60.20         | 32.56      | 61.11     |


|     I\_BS\_01          |  TOTAL (RED)  | FG (RED)  | BG (RED) | TOTAL (GREEN)  | FG (GREEN)  | BG (GREEN) | TOTAL (BLUE)  | FG (BLUE)  | BG (BLUE) |
|------------------------|---------------|-----------|----------|----------------|-------------|------------|---------------|------------|-----------|
| Mean                   | 120.27        | 42.05     | 121.70   | 110.56         | 33.89       | 111.98     | 91.10         | 25.54      |  92.31    |
| Std                    | 42.59         | 37.58     | 41.36    | 41.14          | 26.03       | 40.03      | 39.46         | 21.01      | 38.69     |

|     I\_IL\_01          |  TOTAL (RED)  | FG (RED)  | BG (RED) | TOTAL (GREEN)  | FG (GREEN)  | BG (GREEN) | TOTAL (BLUE)  | FG (BLUE)  | BG (BLUE) |
|------------------------|---------------|-----------|----------|----------------|-------------|------------|---------------|------------|-----------|
| Mean                   |  99.93        | 76.47     | 100.38   | 97.59          | 57.14       | 98.36      | 91.49         | 59.73      | 92.10     |
| Std                    |  62.39        | 49.09     | 62.57    | 60.28          |  37.96      | 60.40      | 57.85         | 32.85      | 58.10     |


##### 5.1.1. Distribution per pixel process

#### 5.2. Data Visualization

### 6. Methodology

#### 6.1. Stauffer and Grimson's algorithm

In section 4 we already discussed the general idea of using GMM based algorithms for image segmentation. However, there as still things we have to think about before can start to implement such an algorithm.

Remember that we are required to fit a GMM for each pixel of an image. In our setting, this would mean that we have to fit 101376 (352x288 pixels) GMM every time we encounter a new frame. Fitting so many Gaussians from scratch takes time and we would never be able to implement it in a way that it satisfied the needs of real-time applications.

To overcome this problem Stauffer and Grimson came up with the idea of updating the existing Gaussian iteratively every time it encounters a new frame. This reduces the time required to fit such Gaussian significantly.

**Gaussian update rule:**

<center>$\mu_t = (1-\rho) \mu_{t-1} + \rho X_t$</center>

<center>$\sigma_t^2 = (1-\rho)\sigma^2_{t-1} + \rho (X_t - \mu)^T (X_t - \mu)$</center>

where $\rho = \alpha \cdot \eta(X_t | \mu_k, \sigma_k)$, $\alpha$ is the learning rate and $X_t$ the new pixel value we observe.

<br/>
**Estimating the background model**

Fitting a GMM is not enough to classify a frame's pixel value as background or foreground pixel value. We still need to determine which Gaussian represents background or foreground pixel value.

Stauffer and Grimson suggest ordering the Gaussians of a mixture model according to their $\omega/\sigma$ ratio. This value increases both as a distribution gains more evidence and as the variance decreases. This ordering of the model is effectively an ordered, open-ended list, where the most likely background distributions remain on top and the less probable transient background distributions gravitate towards the bottom.

Then first $B$ distributions are chosen as background model that account for a predefined
fraction of the evidence $P$.

<center>$B = \text{argmin}_b \sum_{k=1}^b w_k > T$</center>

<br/>
**Classification of pixel values**

As we know which Gaussian best represent background regions, we can now use them to classify new pixel values. A pixel value is considered as "close" to the Gaussian if the value is not more than 2.5 standard deviations away from its mean. If this is the case, we classify the point as background pixel.

Such a proximity measure is the **Mahalanobis distance**. It measures how many standard deviations away a point is from the mean $\mu$.

<center>$D(x) = \sqrt{(x-\mu)^T \Sigma^-1 (x - \mu)}$</center>

### References

**[1]** Cheung S, Kamath C. Robust background subtraction with
foreground validation for Urban Traffic Video. J Appl Signal Proc,
Special Issue on Advances in Intelligent Vision Systems: Methods
and Applications (EURASIP 2005), New York, USA, 2005; 14:
2330-2340  
**[2]** Carranza J, Theobalt C, Magnor M, Seidel H. Free-Viewpoint
Video of Human Actors, ACM Trans on Graphics 2003; 22(3):
569-577  
**[3]** Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. CoRR, May 2015  
**[4]** Stauffer C, Grimson W. Adaptive background mixture models for
real-time tracking. Proc IEEE Conf on Comp Vision and Patt Recog
(CVPR 1999) 1999; 246-252.  
**[5]** http://www.gti.ssr.upm.es/data/LASIESTA  
**[6]** https://en.wikipedia.org/wiki/Precision_and_recall  
**[7]** https://en.wikipedia.org/wiki/F1_score  
**[8]** A Dempster, N. Laird, and D. Rubin. Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society, 39 (Series B):1-38, 1977.  
**[9]** https://github.com/dsoellinger/Background-mixture-models-for-real-time-tracking/blob/master/analysis/analysis\_numerical.ipynb