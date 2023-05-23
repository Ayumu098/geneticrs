from .Population import Population
import matplotlib.pyplot as plt
from typing import List
import numpy


class EvolutionHistory:
    def __init__(self):
        """Stores crossover count, mutation count, and fitness values
        per generation of a Genetic Algorithm.
        """

        self._enabled = True
        self._crossover = []
        self._mutation = []
        self._fitness = []

        self._best_fitness = None
        self._best_population = None

        self._population_size = 0

    def __len__(self):
        return len(self._crossover)

    def __getitem__(self, generation: int):
        return {
            "crossover": self._crossover[generation],
            "mutation": self._mutation[generation],
            "fitness": self._fitness[generation],
        }

    @property
    def is_enabled(self) -> bool:
        """Check if history is allowed to append values."""
        return self._enabled

    @property
    def initial_fitness(self) -> float:
        """Get best fitness of initial generation.

        Returns:
            float: Best fitness of initial generation.
        """
        return self._fitness[0]

    @property
    def latest_fitness(self) -> float:
        """Get best fitness of latest generation.

        Returns:
            float: Best fitness of latest generation.
        """
        return self._fitness[-1]

    @property
    def best_fitness(self) -> float:
        """Get best fitness for all generations.

        Returns:
            float: Best fitness for all generations.
        """
        return self._best_fitness

    @property
    def best_population(self) -> None:
        """Get best population for all generations.

        Returns:
            `torch.tensor`: Best population for all generations.
        """
        return self._best_population

    @property
    def running_best_fitness(self):
        """Returns the best fitness with respect to the past generations
        of the current index.
        """

        return [max(self.fitness[:index]) for index in range(1, len(self) + 1)]

    def freeze(self) -> None:
        """Freeze history to prevent appending of values."""
        self._enabled = False

    def unfreeze(self) -> None:
        """Unfreeze history to allow appending of values."""
        self._enabled = True

    def clear(self) -> None:
        """Clear all history values."""

        self._crossover = []
        self._mutation = []
        self._fitness = []
        self._best_fitness = None
        self._best_population = None

    def update(
        self,
        population: Population,
        crossover: int = None,
        mutation: int = None,
    ) -> None:
        """Append crossover count, mutation count, and fitness values
        of generation to history.

        Args:
            population (`Population`): Population of a generation.

            crossover (`int`): Number of crossovers that occured in a
            generation.
            Defaults to None.

            mutation (`int`): Number of mutations that occured in a generation.
            Defaults to None.
        """
        if self.is_enabled:
            assert crossover >= 0, "Crossover count must be non-negative"
            self._crossover.append(crossover)

            assert mutation >= 0, "Mutation count must be non-negative"
            self._mutation.append(mutation)

            fitness = population.fitness[0][0].item()
            self._fitness.append(fitness)

            if self._best_population is None or fitness > self._best_fitness:
                self._best_population = population.individuals.clone()
                self._best_fitness = fitness

            self._population_size = len(population.individuals)

    def plot(self) -> None:
        """Plots the crossover count, mutation count, and fitness values"""

        # Create generations array
        generations = numpy.array(list(range(len(self))))

        # Plot genetic operations through generations

        plt.subplot(2, 1, 1)
        plt.title("Population Partition")

        mutation = numpy.array(self._mutation)
        crossover = numpy.array(self._crossover)
        selection = numpy.array([self._population_size] * len(self)) - (
            mutation + crossover
        )

        operations = numpy.vstack([mutation, crossover, selection])

        plt.stackplot(
            generations,
            operations,
            labels=[
                f"Mutation ({mutation.sum()})",
                f"Crossover ({crossover.sum()})",
                f"Selection ({selection.sum()})",
            ],
        )

        plt.legend()
        plt.grid()
        plt.ylim((0, 10))
        plt.xlim(0, len(self) - 1)

        # Plot fitness through generations

        plt.subplot(2, 1, 2)
        plt.title("Fitness Scores")

        ones = numpy.ones(len(self))

        fitness = numpy.array(self._fitness)
        best_fitness = numpy.array(self.best_fitness) * ones
        initial_fitness = numpy.array(self.initial_fitness) * ones

        plt.plot(
            generations,
            fitness,
            label="Running Fitness",
        )
        plt.plot(
            generations,
            best_fitness,
            label=f"Best Fitness ({self.best_fitness})",
            linestyle="dashed",
        )
        plt.plot(
            generations,
            initial_fitness,
            label=f"Initial Fitness ({self.initial_fitness})",
            linestyle="dashed",
        )

        plt.legend()
        plt.grid()
        plt.ylim((min(self.fitness) * 0.9, max(self.fitness) * 1.1))
        plt.xlim(0, len(self) - 1)

        plt.show()

    @property
    def crossover(self) -> List[int]:
        """_summary_

        Returns:
            List[int]: Crossover count of population across generations
        """
        return self._crossover

    @property
    def mutation(self) -> List[int]:
        """
        Returns:
            List[int]: Mutation count of population across generations
        """
        return self._mutation

    @property
    def fitness(self) -> list[list[float]]:
        """
        Returns:
            List: Fitness scores of population across generations
        """
        return self._fitness
