from ...model.GeneticAlgorithm import Population
from random import randint
import torch


def random_parameters() -> tuple[int, int]:
    """
    Returns:
        `Tuple[int, int]`: Random population, and gene size
    """
    return (
        torch.randint(1, 1000, (1,)).item(),
        torch.randint(1, 1000, (1,)).item(),
    )


def test_population_shape() -> None:
    """Test population shape and size."""

    parameters = [random_parameters() for _ in range(randint(1, 100))]

    gene_bound = 1000
    for size, gene_size in parameters:
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

    gene_bound = 1000
    for size, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

    assert population.selection().size(dim=0) == gene_size


def test_mutation() -> None:
    """Test if mutation produces a tensor of the same size as the gene size."""
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    gene_bound = 1000
    for size, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

    assert population.mutation_produce().size(dim=0) == gene_size


def test_crossover() -> None:
    """Test if crossover produces a tensor of the same size as the gene size."""
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    gene_bound = 1000
    for size, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

    assert population.crossover_produce().size(dim=0) == gene_size


def test_set() -> None:
    """Test if population can be set."""

    parameters = [random_parameters() for _ in range(randint(1, 100))]

    gene_bound = 1000
    for size, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

        random_index = randint(0, population.size - 1)
        new_individual = population.mutation_produce()

        initial_individual = population[random_index].clone()
        population[random_index] = new_individual

        assert not population[random_index].equal(initial_individual)


def test_clear() -> None:
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    gene_bound = 1000
    for size, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

        previous_individuals = population.individuals.clone()
        population.clear()

        assert not population.individuals.equal(previous_individuals)


def test_best_worst_individuals():
    size, gene_size, gene_bound = 10, 10, 1000
    population = Population(
        size=size,
        gene_size=gene_size,
        gene_bound=gene_bound,
    )

    for sample in range(1, size):
        bests, weaks = sample, size - sample

        for best_individual in population.best_individuals(bests):
            for weak_individual in population.worst_individuals(weaks):
                assert best_individual.sum() >= weak_individual.sum()


def test_sort():
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    gene_bound = 1000
    for size, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

        population.sort()

        for index in range(1, population.size):
            assert population[index - 1].sum() >= population[index].sum()


def test_drop():
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    gene_bound = 1000
    for size, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

        keep = torch.randint(1, population.size, (1,)).item()
        population.drop(keep)

        assert population.size == keep


def test_append():
    parameters = [random_parameters() for _ in range(randint(1, 100))]

    gene_bound = 1000
    for size, gene_size in parameters:
        population = Population(
            size=size,
            gene_size=gene_size,
            gene_bound=gene_bound,
        )

        new_individuals = torch.randint(gene_bound, (size, gene_size))
        population.append(new_individuals)

        assert population.size == size * 2
