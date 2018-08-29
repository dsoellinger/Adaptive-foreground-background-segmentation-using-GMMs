from .data_loader import DataLoader
import os
from scipy.misc import imread
import sys


class LASIESTADataLoader(DataLoader):

    def __init__(self, original_path, label_path=None, name=None):

        self._original_path = original_path
        self._label_path = label_path
        self._next_frame_idx = 0
        self._name = name

        original_files = os.listdir(original_path)
        original_files.sort(key=lambda file: int(file[file.find('-') + 1:file.find('.')]))
        self._original_files = original_files

        if self._label_path is not None:
            label_files = os.listdir(label_path)
            label_files.sort(key=lambda file: int(file[file.rfind('_') + 1:file.find('.')]))
            self._label_files = label_files

            if len(self._original_files) != len(self._label_files):
                sys.exit(-1)

    def get_next_frame(self):

        if self._next_frame_idx < len(self._original_files):

            path_to_original_frame = os.path.join(self._original_path, self._original_files[self._next_frame_idx])
            original_frame = imread(path_to_original_frame)

            if self._label_path is not None:
                path_to_label_frame = os.path.join(self._label_path, self._label_files[self._next_frame_idx])
                label_frame = imread(path_to_label_frame)
                self._next_frame_idx += 1
                return original_frame, label_frame

            self._next_frame_idx += 1

            return original_frame

        return None

    def set_next_frame_idx(self, idx):
        self._next_frame_idx = idx

    def get_next_frame_idx(self):
        return self._next_frame_idx

    def get_nr_of_frames(self):
        return len(self._original_files)

    def get_name(self):
        return self._name

    def __iter__(self):
        return self

    def __next__(self):

        frame = self.get_next_frame()

        if frame is None:
            raise StopIteration()

        return frame
