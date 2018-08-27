from segmentizer import Segmentizer
from segmentizer.data_loader import LASIESTADataLoader
import time

data_loader = LASIESTADataLoader('/Users/dsoellinger/Downloads/I_SI_01')
video_segmentizer = Segmentizer(352,288)

start = time.time()

for i, frame in enumerate(data_loader):

    if i == 10:
        break

    print("Frame: " + str(i + 1))
    video_segmentizer.fit_and_predict(frame)

end = time.time()

print("Elapsed time: " + str(end-start))