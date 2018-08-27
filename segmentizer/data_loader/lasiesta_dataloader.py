from .data_loader import DataLoader
import os
from scipy.misc import imread
import sys


class LASIESTADataLoader(DataLoader):

    def __init__(self, groundtruth_path, label_path=None):

        self._groundtruth_path = groundtruth_path
        self._label_path = label_path
        self._next_frame_idx = 0

        # Load groundtruth
        groundtruth_files = os.listdir(groundtruth_path)
        groundtruth_files.sort(key=lambda file: int(file[file.find('-') + 1:file.find('.')]))
        self._groundtruth_frames = groundtruth_files

        if self._label_path is not None:
            label_files = os.listdir(label_path)
            label_files.sort(key=lambda file: int(file[file.rfind('_') + 1:file.find('.')]))
            self._label_frames = label_files

            if len(self._groundtruth_frames) != len(self._label_frames):
                sys.exit(-1)

    def get_next_frame(self):

        if self._next_frame_idx < len(self._groundtruth_frames):

            path_to_groundtruth_frame = os.path.join(self._groundtruth_path, self._groundtruth_frames[self._next_frame_idx])
            groundtruth_frame = imread(path_to_groundtruth_frame)

            self._next_frame_idx += 1

            if self._label_path is not None:
                path_to_label_frame = os.path.join(self._label_path, self._label_frames[self._next_frame_idx])
                label_frame = imread(path_to_label_frame)
                return groundtruth_frame, label_frame

            return groundtruth_frame

        return None

    def set_next_frame_idx(self, idx):
        self._next_frame_idx = idx

    def get_next_frame_idx(self):
        return self._next_frame_idx

    def get_nr_of_frames(self):
        return len(self._frames)

    def __iter__(self):
        return self

    def __next__(self):

        frame = self.get_next_frame()

        if frame is None:
            raise StopIteration()

        return frame
