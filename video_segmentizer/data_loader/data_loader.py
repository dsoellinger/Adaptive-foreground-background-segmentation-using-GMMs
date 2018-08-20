from abc import ABC, abstractmethod


class DataLoader(ABC):

    @abstractmethod
    def get_next_frame(self):
        pass
