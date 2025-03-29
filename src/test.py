from enum import Enum

class EMG(Enum):
    A = 1
    B = 2
    C = 3


def getElement(map, emg: EMG):
    return map[emg.name]


map = {"A": "test", "B": 2, "C": 3}

def getElem(emg: EMG):
    return map[emg.name]
print(getElement(map, EMG.A))