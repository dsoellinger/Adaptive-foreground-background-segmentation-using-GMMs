# Machine Learning Engineer Nanodegree

## Capstone Proposal
Dominik Söllinger  
August 17th, 2018

## Foreground-background segmentation using Gaussian mixture models 

### Domain Background

Background subtraction is a common pre-processing step in many computer vision related tasks. Many applications [1][2] only require information of changes in a scene since such region typically contains the objects we are interest in. For instance, cars that drive on highway or pedestrians the walk on a sidewalk. Background segmentation allows us to detect such objects by a pixel-wise segmentation of each frames into a foreground and background region.

Deep learning enthusiast might now suggest the use of novel neural networks like UNET [3] to solve such segmentation tasks. However, there are well-studied algorithms like the one proposed by Stauffer and Grimson [4] that provide am unsupervised learning based solution based on Gaussian mixture models (GMMs) to the segmentation problem. The fact that these GMM based algorithms are well-studied and don't require extensive training data strongly advocate its use for foreground-background segmentation problems.

This is also the reason why I want to implement the GMM based segmentation approach suggested by Stauffer and Grimson myself. So far, I only got in touch with neural network based segmentation approaches. However, it's inspiring that same can be done using a relative simple GMM based solution without requiring a labeled dataset at all. Since I haven't seen any code that implements, explains and demonstrates this algorithm in Python it would be serve as a great capstone project and hopefully help others to get a better intuition of how this algorithm works.


### Problem Statement

Goal of this project is to develop a solution for unsupervised foreground-background video segmentation and assess its performance. For performance evaluating we will the model's performance on the LASIESTA [5] dataset. A good model should finally be able to segment any video into a foreground or background region after seeing a few frames. Ideally, the model should be able to perform this segmentation task in real-time.


### Datasets and Inputs

We will use the LASIESTA [5] dataset to train and test the implemented model. The dataset comprises of eight different scenes captured in different indoor and outdoor environments. These environments cover a broad spectrum of scenarios we may encounter in real-world meaning that we have to deal with illumination changes, occlusion and shadows.  
In each scenario we are given frames of the original image (24bpp BMP) as well as the corresponding labels (ground truth) for every frame.  

Label data are given as images where every pixel value uniquely assigns the pixel to a segment:

- Black pixels (0,0,0): Background.
- Red pixels (255,0,0): Moving object with label 1 along the sequence.
- Green pixels (0,255,0): Moving object with label 2 along the sequence.
- Yellow pixels (255,255,0): Moving object with label 3 along the sequence.
- White pixels (255,255,255): Moving objects remaining static.
- Gray pixels (128,128,128): Uncertainty pixels.


### Solution Statement

Various research papers propose the use of Gaussian mixture models for foreground-background segmentation. In the project we want to take a closer look at the approach described by Stauffer and Grimson in [4].

The idea is to fit a Gaussian mixture model to a time series of image pixels. In other words, we are given video and take a certain number of frames out of this video. Next, we consider what's called a "pixel process". The "pixel process" is a time series of pixel values, for example, scalars for gray values or vectors for color images. 

At any time $t$ we know the history (intensity values) of a certain pixel:

<center>$\{X_1,...,X_t\} = \{\hspace{0.2cm} I(x,y,i): 1 \leq i \leq t \hspace{0.2cm} \} \hspace{1cm}$ where $I$ is the image sequence</center>

We can now use this time series to estimate the probability of seeing a certain pixel value. Formally, this means that we can estimate the probability of a certain pixel value by fitting a GMM on the recent history of pixel values.

The probability of observing the current pixel value is 

<center>$P(x) = \sum_{I=1}^K w_{i,t} \cdot \eta(X_t,\mu_{i,t},\Sigma_{i,t})$</center>

where K is the number of distributions, $w_{i,t}$ is an estimate of the weight (what portion of the data is accounted for by this Gaussian) of the i-th Gaussian in the mixture at time $t$, $\mu_{i,t}$ is the mean value of the i-th Gaussian in the mixture at time $t$, $\Sigma_{i,t}$ is the covariance
matrix of the i-th Gaussian in the mixture at time $t$, and where $\eta$ is a Gaussian probability density function.

<center> $\eta(X_t,\mu_{i,t},\Sigma_{i,t}) = \frac{1}{ (2\pi)^{\frac{n}{2}} |\Sigma|^{\frac{1}{2}}}\cdot e^{-\frac{1}{2} (X_t-\mu_t)^T \Sigma^{-1} (X_t-\mu_t)}$ </center>

**Note:** $K$ depends mainly on the modality of the background distribution, but for implementation purposes factors like the available
computational power and real time requirements have to be considered. In practice it has been shown that 3 to 5 is a reasonable choice for $K$.

#### Background Model estimation

The distribution of recently observed values of each pixel in the scene is now characterized by a mixture of Gaussians. This mixture can now be used to estimate the probability that a certain pixel value belongs to a background or foreground regions. The idea is similar to an algorithm referred as Bog-of-words (BoW) classification. Some Gaussians are more likely to represent a background region than others. If we knew which Gaussians represent background objects, we could assign new pixel values to background / foreground by calculating its proximity to background Gaussians.

To understand this, consider the accumulation of supporting evidence and the relatively low
variance for the "background" distributions when a static, persistent object is visible. In contrast, when a new object occludes the background object, it will not, in general, match one of the existing distributions which will result in either the creation of a new distribution
or the increase in the variance of an existing distribution. Also, the variance of the moving object
is expected to remain larger than a background pixel until the moving object stops. [4]

To perform this classification for a pixel we order the Gaussians by the value of $\omega/\sigma$. This value increases both as a distribution gains more evidence and as the variance decreases. This ordering of the model is effectively an ordered, open-ended list, where the most likely background
distributions remain on top and the less probable transient background distributions gravitate towards the bottom.

Now, the first $B$ distributions are chosen as background model that account for a predefined
fraction of the evidence $P$.

<center>$B = \text{argmin}_b \sum_{k=1}^b w_k > T$</center>

#### Classification of pixel values

As we know which Gaussian best represent background regions, we can now use them to classify new pixel values. A pixel value is considered as "close" to the Gaussian if the value is not more than 2.5 standard deviations away from its mean. If this is the case, we classify the point as background pixel.


### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

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