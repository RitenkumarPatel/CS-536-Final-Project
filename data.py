from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np


class Event(ABC):
    @abstractmethod
    def value(self, data: "Data") -> np.ndarray:
        pass


class Data:
    def __init__(self, n: int, rng: np.random.Generator):
        self.n = n
        self.rng = rng
        self.data = dict[Event, np.ndarray]()

    def get_single(self, event: Event):
        if event not in self.data:
            self.data[event] = event.value(self)
        return self.data[event]

    def get(self, events: Sequence[Event]) -> np.ndarray:
        """
        Return matrix, where row is event and col is n (data count)
        """
        return np.array([self.get_single(event) for event in events])


class LinearEvent(Event):
    def __init__(self, m: float, b: float):
        self.m = m
        self.b = b

    def value(self, data: Data) -> np.ndarray:
        x = data.rng.standard_normal(data.n)
        return self.m * x + self.b


class CompositeEvent(Event):
    def __init__(self, events: Sequence[Event], weights: np.ndarray):
        self.events = events
        self.weights = weights

    def value(self, data: Data) -> np.ndarray:
        matrix = data.get(self.events)
        return self.weights @ matrix


class Case(ABC):
    @abstractmethod
    def generate_data(self):
        pass


@dataclass
class CaseVFL(Case):
    rng: np.random.Generator
    n: int
    events: int
    output_arity: int
    sensors: int
    sensor_arity: int
    noise_std: float

    def __post_init__(self):
        self.input_events = [
            LinearEvent(m, b)
            for m, b in self.rng.uniform(-10, 10, size=(self.events, 2))
        ]
        self.output_event = CompositeEvent(
            [
                self.input_events[i]
                for i in self.rng.choice(
                    len(self.input_events), size=self.output_arity, replace=False
                )
            ],
            self.rng.standard_normal(self.output_arity),
        )

    def generate_data(self):
        data = Data(self.n, self.rng)

        X = [
            data.get(
                self.input_events[i * self.sensor_arity : (i + 1) * self.sensor_arity]
            )
            for i in range(self.sensors)
        ]
        Y = data.get_single(self.output_event) + self.rng.normal(
            0, self.noise_std, self.n
        )

        return X, Y
