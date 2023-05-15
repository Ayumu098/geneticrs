from .Population import Population
from typing import List


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
