from abc import ABC, abstractmethod


class Dataset_Builder(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def download_prepare(self):
        pass

    @abstractmethod
    def as_dataframe(self):
        pass