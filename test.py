from segmentizer import Segmentizer
from segmentizer.data_loader import LASIESTADataLoader

video_segmentizer = Segmentizer(352, 288)

data_loader = LASIESTADataLoader('/Users/dsoellinger/Downloads/I_SI_01')

frame = data_loader.get_next_frame()

nr_of_frames = 20
frame_idx = 0

while frame is not None:
    video_segmentizer.fit(frame)
    background_image = video_segmentizer.classify_image(frame)
    frame = data_loader.get_next_frame()
    frame_idx += 1

    if frame_idx >= nr_of_frames:
        break

print("Done")
