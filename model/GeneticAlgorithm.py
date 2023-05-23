import torch
from .Population import Population
from .EvolutionHistory import EvolutionHistory


class GeneticAlgorithm:
    def __init__(
        self,
        population: Population = Population(),
        history: EvolutionHistory = EvolutionHistory(),
        mutation_probability: float = 0.1,
        crossover_probability: float = 0.3,
    ) -> None:
        """Class wrapper for Genetic Algorithm

        Args:
            population (`Population`):
            Collection of individuals in a standard genetic algorithm.

            history (`EvolutionHistory`):
            Tracks the evolution of the population.

            mutation_probability (`float`, optional):
            Probability of mutation in evolution.
            Defaults to 0.1.

            crossover_probability (`float`, optional):
            Probability of crossover in evolution.
            Defaults to 0.3.
        """

        self._population = population
        self._history = history

        # Genetic Operator Parameters

        assert 0 <= mutation_probability <= 1.0
        assert 0 <= crossover_probability <= 1.0
        assert mutation_probability + crossover_probability <= 1.0

        self._mutation_probability = mutation_probability
        self._crossover_probability = crossover_probability

        nonselection_probability = mutation_probability + crossover_probability
        self._selection_probability = 1 - nonselection_probability

        self.history.update(self.population, crossover=0, mutation=0)

    def __iter__(self):
        return iter(self.population)

    def __getitem__(self, index: int):
        return self.population[index]

    @property
    def population(self):
        """
        Returns:
            `Population`:
             Collection of individuals in a standard genetic algorithm.
        """
        return self._population

    @property
    def history(self):
        """
        Returns:
            `EvolutionHistory`:
            Tracks the evolution of the population.
        """
        return self._history

    @property
    def mutation_probability(self):
        """
        Returns:
            `float`:
            Probability of mutation in evolution.
        """

        return self._mutation_probability

    @property
    def crossover_probability(self):
        """
        Returns:
            `float`:
            Probability of crossover in evolution.
        """

        return self._crossover_probability

    @property
    def selection_probability(self):
        """
        Returns:
            `float`:
            Probability of selection in evolution.
        """

        return self._selection_probability

    def operation_partition(self) -> tuple[int]:
        """Creates a random partitions with values [0, 1, 2] that represent
        the genetic operations to perform on the population.
            0: Selection
            1: Crossover
            2: Mutation
        Returns:
            `torch.tensor`: Genetic operations to perform on the population
        """
        mutation_probability = self.mutation_probability
        crossover_probability = self.crossover_probability
        selection_probability = self.selection_probability

        operation_probability = torch.tensor(
            [
                selection_probability,
                mutation_probability,
                crossover_probability,
            ]
        )

        operation_partition = operation_probability.float().multinomial(
            self.population.size, replacement=True
        )

        return operation_partition

    def evolve(self, generations: int = 1) -> None:
        """Applies crossover, mutation and fitness function to population.

        Args:
            generation (`int`, optional): Number of runs. Defaults to 1.
        """

        for _ in range(generations):
            # Generate random partition of operations
            
            operation_partition = self.operation_partition().sort()[0]
            operation_map = [
                self.population.selection,
                self.population.crossover_produce,
                self.population.mutation_produce,
            ]

            new_population = []

            # Genereate new population
            for index, operation in enumerate(operation_partition):
                genetic_operation = operation_map[operation]
                new_population[index] = genetic_operation()

            # Overwrite old population
            for index, new_individual in enumerate(new_population):
                self.population[index] = new_individual

            # Update history
            mutation_count = (operation_partition == 1).sum()
            crossover_count = (operation_partition == 2).sum()

            self.history.update(
                population=self.population,
                crossover=crossover_count,
                mutation=mutation_count,
            )

    def clear(self) -> None:
        """Resets the population to a random binary state. Clears history."""
        self.population.clear()
        self.history.clear()
