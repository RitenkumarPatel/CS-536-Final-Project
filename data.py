from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np


# Abstraction for a generic random variable
class Event(ABC):
    @abstractmethod
    def value(self, data: "Data") -> np.ndarray:
        pass


# Container to hold samples, given some "definition" of a random variable
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


# Gaussian event
class LinearEvent(Event):
    def __init__(self, m: float, b: float):
        self.m = m
        self.b = b

    def value(self, data: Data) -> np.ndarray:
        x = data.rng.standard_normal(data.n)
        return self.m * x + self.b


# Linear combination of some events
class CompositeEvent(Event):
    def __init__(self, events: Sequence[Event], weights: np.ndarray):
        self.events = events
        self.weights = weights

    def value(self, data: Data) -> np.ndarray:
        matrix = data.get(self.events)
        return self.weights @ matrix


# Generic class for different test cases.
# Initialization randomly sets parameters
# generate_data should sample from the same distribution each call!
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


@dataclass
class CaseVFLCorrelated(Case):
    rng: np.random.Generator
    n: int
    events: int
    output_arity: int
    parties: int
    party_arity: int

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

        self.party_indices = [
            [(i * self.party_arity + j) % self.events for j in range(self.party_arity)]
            for i in range(self.parties)
        ]

    def generate_data(self):
        data = Data(self.n, self.rng)

        X = [
            data.get([self.input_events[j] for j in self.party_indices[i]])
            for i in range(self.parties)
        ]
        Y = data.get_single(self.output_event) + self.rng.normal(
            0, self.noise_std, self.n
        )

        return X, Y


@dataclass
class CaseHFL(Case):
    rng: np.random.Generator
    n: int
    events: int
    output_arity: int
    sensors: int
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

        X_global = data.get(self.input_events)

        Y_global = data.get_single(self.output_event) + self.rng.normal(
            0, self.noise_std, self.n
        )

        X = np.array_split(X_global, self.sensors, axis=1)
        Y = np.array_split(Y_global, self.sensors)

        return X, Y


@dataclass
class CaseHFL_Heterogeneous:
    rng: np.random.Generator
    n: int
    events: int
    output_arity: int
    sensors: int
    noise_std: float
    heterogeneity_std: float = 1.0

    def __post_init__(self):
        self.input_events = [
            LinearEvent(m, b)
            for m, b in self.rng.uniform(-10, 10, size=(self.events, 2))
        ]

        active_indices = self.rng.choice(
            len(self.input_events), size=self.output_arity, replace=False
        )
        active_events = [self.input_events[i] for i in active_indices]

        base_weights = self.rng.standard_normal(self.output_arity)

        self.sensor_output_events = []
        for _ in range(self.sensors):
            local_weights = base_weights + self.rng.normal(
                0, self.heterogeneity_std, self.output_arity
            )
            self.sensor_output_events.append(
                CompositeEvent(active_events, local_weights)
            )

    def generate_data(self):
        X_hetero = []
        Y_hetero = []

        if self.n % self.sensors != 0:
            raise ValueError(
                f"Total samples (n={self.n}) must be perfectly divisible by the "
                f"number of sensors ({self.sensors}) to ensure balanced heterogeneous data."
            )
        num_samples_per_sensor = self.n // self.sensors

        for i in range(self.sensors):
            local_data = Data(num_samples_per_sensor, self.rng)
            X_local = local_data.get(self.input_events)

            Y_local = local_data.get_single(
                self.sensor_output_events[i]
            ) + self.rng.normal(0, self.noise_std, num_samples_per_sensor)

            X_hetero.append(X_local)
            Y_hetero.append(Y_local)

        return X_hetero, Y_hetero
