from abc import ABC, abstractmethod
from collections.abc import Sequence
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


def case1(
    n: int, events: int, outputs: int, output_arity: int, nodes: int, node_arity: int
):
    rng = np.random.default_rng(0)
    data1 = Data(n, rng)

    linear_events = [
        LinearEvent(m, b) for m, b in rng.uniform(-10, 10, size=(events, 2))
    ]

    output_events = [
        CompositeEvent(
            [
                linear_events[i]
                for i in rng.choice(
                    len(linear_events), size=output_arity, replace=False
                )
            ],
            rng.standard_normal(output_arity),
        )
        for _ in range(outputs)
    ]
