import torch
from .GeneticAlgorithm import GeneticAlgorithm


def random_int() -> int:
    """
    Returns:
        `int`: Random integer from 1 to 100
    """
    lower_limit, upper_limit = 1, 100
    return torch.randint(lower_limit, upper_limit, (1,)).item()


def random_parameters() -> tuple[int, int, int]:
    """
    Returns:
        `Tuple[int, int, int]`: Random population, gene code max, and gene size
    """
    return random_int(), random_int(), random_int()


def random_probabilities() -> tuple[float, float]:
    """
    Returns:
        `Tuple[float, float]`: Random crossover and mutation probabilities
    """

    mutation_probability = torch.rand(1).item() % 1
    crossover_probability = (torch.rand(1).item() % 1) % (1 - mutation_probability)
    return mutation_probability, crossover_probability


def test_evolve() -> None:
    """Test if fitness increases overtime"""

    solver = GeneticAlgorithm()
    initial_individuals = solver.population.individuals.clone()

    solver.evolve(random_int())
    current_individuals = solver.population.individuals.clone()

    assert not initial_individuals.equal(current_individuals)
