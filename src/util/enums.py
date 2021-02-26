from enum import Enum, auto


class DataSplit(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class FrequencyMetric:
    EPOCHS = auto()
    ITERATIONS = auto()
