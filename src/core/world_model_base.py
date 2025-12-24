"""
Base mode of world model
"""
from abc import ABC, abstractmethod

class WorldModelBase(ABC):
    @abstractmethod
    def add_experiment(self, experiment):
        pass

    @abstractmethod
    def count_experiments(self) -> int:
        pass

    @abstractmethod
    def summarize(self) -> dict:
        """Return a summary of the current world state"""
        pass

    @abstractmethod
    def update_with_insights(self, insights):
        pass
