from .GeneticAlgorithm import Population
from random import randint


def random_parameters() -> tuple[int, int, int]:
    """
    Returns:
        `Tuple[int, int, int]`: Random population, gene code max, and gene size
    """
    return randint(1, 1000), randint(1, 1000), randint(1, 1000)


def test_population_shape() -> None:
    """Test population shape and size."""

    parameters = [random_parameters() for _ in range(randint(1, 100))]

    for size, gene_bound, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

        assert population.gene_size == gene_size
        assert population.size == size
        assert population.shape == (size, gene_size)


def test_selection() -> None:
    """Test if selection produces a tensor of the same size as the gene size."""
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    for size, gene_bound, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

    assert population.selection().size(dim=0) == gene_size


def test_mutation() -> None:
    """Test if mutation produces a tensor of the same size as the gene size."""
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    for size, gene_bound, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

    assert population.mutation_produce().size(dim=0) == gene_size


def test_crossover() -> None:
    """Test if crossover produces a tensor of the same size as the gene size."""
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    for size, gene_bound, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

    assert population.crossover_produce().size(dim=0) == gene_size


def test_set() -> None:
    """Test if population can be set."""

    parameters = [random_parameters() for _ in range(randint(1, 100))]

    for size, gene_bound, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

        random_index = randint(0, population.size-1)
        new_individual = population.mutation_produce()

        initial_individual = population[random_index].clone()
        population[random_index] = new_individual

        assert not population[random_index].equal(initial_individual)


def test_clear() -> None:
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    for size, _, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=1000,
        )

        previous_individuals = population.individuals.clone()
        population.clear()

        assert not population.individuals.equal(previous_individuals)
