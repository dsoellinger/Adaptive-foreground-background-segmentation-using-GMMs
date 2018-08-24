from segmentizer import Segmentizer
from segmentizer.data_loader import LASIESTADataLoader

video_segmentizer = Segmentizer(352,288)

data_loader = LASIESTADataLoader('/Users/dsoellinger/Downloads/I_SI_01')

frame = data_loader.get_next_frame()

while frame is not None:
    background_image = video_segmentizer.classify_image(frame)
    video_segmentizer.fit(frame)
    frame = data_loader.get_next_frame()
    
print("Done")
