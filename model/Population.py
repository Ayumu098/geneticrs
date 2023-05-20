import torch


class Population:
    @staticmethod
    def random_binary(size: tuple[int] = (1,)):
        """
        Args:
            size (`tuple[int]`): Size of random binary matrix to generate

        Returns:
            `torch.tensor`: Random Binary Matrix
        """
        return torch.bernoulli(torch.rand(size))

    def __init__(
        self,
        size: int = 10,
        gene_size: int = 20,
        gene_bound: int = 1000,
        fitness_function: callable = lambda x: x.sum(dim=1),
        device="cpu",
    ) -> None:
        """Collection of individuals in a standard genetic algorithm.

        Args:
            size (`int`, optional):
            Number of individuals.
            Defaults to 10.

            gene_size (`int`, optional):
            Number of characteristics for an individual.
            Defaults to 20.

            gene_bound (`int`, optional):
            Highest integer value for a characteristic.
            Defaults to 10_000.

            fitness_function (`callable`, optional):
            Function that returns tensor[population] of fitness scores.
            Defaults to the sum of the individual's genes.

            device (str, optional):
            Device to load tensor(population, gene) for individuals.
            Defaults to "cpu".
        """

        # Population shape

        assert gene_size > 0
        assert size > 0

        self._size = size
        self._gene_size = gene_size
        self._gene_bound = gene_bound
        self._shape = (size, gene_size)

        # Fitness Function
        self._fitness_function = fitness_function

        # Population Initialization
        self._individuals = torch.randint(self.gene_bound, self.shape)
        self._individuals.to(device)
        self._device = device

    def __iter__(self):
        return iter(self._individuals)

    def __getitem__(self, index: int):
        return self._individuals[index]

    def __setitem__(self, index: int, value: torch.tensor) -> None:
        """Sets an individual in the population
        Args:
            index (`int`): Index of individual to set
            value (`torch.tensor`): Individual to set
        """
        assert value.shape == (self.gene_size,)

        self._individuals[index] = value

    @property
    def gene_size(self) -> int:
        """
        Returns:
            `int`: Fixed number of integer-coded characteristics per individual
        """
        return self._gene_size

    @property
    def gene_bound(self) -> int:
        """
        Returns:
            int: Upper bound of integer-coded characteristics
        """
        return self._gene_bound

    @property
    def shape(self) -> tuple[int]:
        """
        Returns:
            `int`: `population_size`,
            `int`: `gene_size`
        """
        return self._shape

    @property
    def size(self) -> tuple[int]:
        """
        Returns:
            `int`: `population_size`,
        """
        return self._size

    @property
    def fitness(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the provided `fitness_function` on the population.

        Returns:
            `torch.tensor(population size)`: Sorted fitness scores

            `torch.tensor(population size)`: Sorted indices from population
        """
        return self._fitness_function(self._individuals).sort(descending=True)

    @property
    def individuals(self):
        """
        Returns:
            `torch.tensor(population_size, gene_size)`:
             Rows represents individuals with integer-coded characteristics
        """
        return self._individuals

    def selection(self) -> torch.tensor:
        """Selects individuals from the population based on fitness scores

        Returns:
            `torch.tensor(population_size, gene_size)`:
            Selected individuals from the population
        """
        selection_index = self.fitness[0].float().multinomial(1)
        return self._individuals[selection_index][0]

    def crossover_produce(self) -> torch.tensor:
        """Creates a new individual from two individuals in the population

        Returns:
            `torch.tensor`:
            New individual from crossover
        """
        crossover_points = self.random_binary((self.gene_size,))
        individual, other = self.selection(), self.selection()
        new_individual = individual

        for point, crossover_point in enumerate(crossover_points):
            if crossover_point:
                new_individual[point] = other[point]

        return new_individual

    def mutation_produce(self) -> torch.tensor:
        """Mutates an individual in the population

        Returns:
            `torch.tensor`:
            New individual from mutation
        """
        mutation_points = self.random_binary((self.gene_size,))
        new_individual = self.selection()

        for index, mutation_point in enumerate(mutation_points):
            if mutation_point:
                new_individual[index] = torch.randint(self.gene_bound, (1,))

        return new_individual

    def clear(self) -> None:
        """Resets population to random individuals."""
        self._individuals = torch.randint(self.gene_bound, self.shape)
        self._individuals.to(self._device)
