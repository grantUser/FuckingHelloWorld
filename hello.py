import os
import string

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class print:
    def __new__(cls, *args) -> None:
        parts = [
            str(arg)
            if isinstance(arg, str)
            else (repr(arg) if hasattr(arg, "__repr__") else None)
            for arg in args
        ]
        output = " ".join(parts) + "\n"
        os.write(1, output.encode())


class Base:
    characters = (
        string.ascii_letters + string.digits + string.punctuation + string.whitespace
    )

    def train(self):
        self.model.fit(self.x_train.reshape(-1, 1), self.y_train)


class BINARYAgent(Base):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.binary_values = [bin(ord(char))[2:].zfill(8) for char in self.characters]
        self.x_train = np.array([int(binary, 2) for binary in self.binary_values])
        self.y_train = np.array([ord(char) for char in self.characters])

        self.train()

    def predict(self, value=None):
        int_value = int(value.zfill(8), 2)
        predicted_index = self.model.predict(np.array([[int_value]]))
        return int(predicted_index[0])


class ASCIIAgent(Base):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.x_train = np.array([ord(char) for char in self.characters]).reshape(-1, 1)
        self.y_train = np.arange(len(self.characters))

        self.train()

    def predict(self, value=None):
        if isinstance(value, int):
            prediction = self.model.predict([[value]])
            if prediction:
                return self.characters[int(prediction[0])]
        return None


class FuckingHelloWorld:
    def __init__(self) -> None:
        self.ascii = ASCIIAgent()
        self.binary = BINARYAgent()

        self.__post_init__()

    def __post_init__(self):
        binary_strings = [
            "01001000", "01100101", "01101100",
            "01101100", "01101111", "00100000",
            "01010111", "01101111", "01110010",
            "01101100", "01100100", "00100001",
        ]

        message = "".join(
            self.ascii.predict(self.binary.predict(binary_string))
            for binary_string in binary_strings
        )
        print(message)


FuckingHelloWorld()
