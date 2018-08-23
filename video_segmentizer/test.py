from data_loader import LASIESTADataLoader
from model import IIDGaussian, PixelProcess, VideoSegmentizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

video_segmentizer = VideoSegmentizer(352,288)

data_loader = LASIESTADataLoader('/Users/dsoellinger/Downloads/I_SI_01')

frame = data_loader.get_next_frame()

video_segmentizer.fit(frame)
    
print("Done")
