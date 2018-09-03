# Adaptive foreground-background segmentation using GMMs


This project illustrates how to apply adaptive background-segmentation for videos.  
More precisely, I implement the method by proposed by Stauffer and Grimson in their paper "Adaptive background mixture models for real-time tracking" [1]. Performance evaluation was done based on different indoor scenes from the LASIESTA [2] dataset.

This project was also my capstone project for the Udacity Machine Learning Nanodegree. So, if you want interested in a more detailed explanation of the project, please checkout out my project report.

**Note:** The developed Python code does not allow to segment videos in real-time. An efficient real-time implementation needs to be done in C / C++. However, it's a great starting point if you just want to see and understand the basic concept.


<p float="left">
  <img src="https://raw.githubusercontent.com/dsoellinger/Background-mixture-models-for-real-time-tracking/master/submission/images/sample_seg_1.gif" width="400px" style="margin-right: 10px;"/>
  <img src="https://raw.githubusercontent.com/dsoellinger/Background-mixture-models-for-real-time-tracking/master/submission/images/sample_seg_2.gif" width="400px" style="margin-left: 10px;"/> 
</p>

### How to segment our own videos?

You got your own video you want to segment? This is easy.  
I've built a small library that takes individual video frames as input and returns a segmented frame as output. To install and use the library simple execute the following steps:

1. **Check or install dependencies**
	
	Run the code please make sure that you have **Numpy** and **Numba** installed on our system. **Numba** speeds up required matrix computations by means of just-in-time compilation.

2. **Install `segmentizer` library**  

	Run `python3 setup.py install` 

	**Note:** setup.py can be found inside the folder "framework"

3. **Fit and train our own model**  

	```
	from segmentizer import Segmentizer
		
	segmentizer = Segmentizer(frame_width, frame_height)
	segmented_frame = segmentizer.fit_and_predict(our_frame)
	```
	
	**Note:** To segment the whole video simply path all frames to `fit_and_predict` iteratively. The method returns a 2D Python list object with binary values where
	- **True:** Background pixel
	- **False:** Foreground pixel


### The implementation

If you are curious and you want to see the actual implementation, I recommend to take a look at the classes `RGBPixelProcess` and `IIDGaussian`.

### Good to know

The original paper is not very detailed in terms of the original implementation. I implemented the code based on my own understanding of the paper and therefore it might not coincide with the one from the original paper.

### References

**[1]** Stauffer C, Grimson W. Adaptive background mixture models for
real-time tracking. Proc IEEE Conf on Comp Vision and Pattern Recognition (CVPR 1999) 1999; 246-252.  
**[2]** http://www.gti.ssr.upm.es/data/LASIESTA  
