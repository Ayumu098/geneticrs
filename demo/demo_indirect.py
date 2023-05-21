"""
Demonstrate the general algorithm using an index-based fitness function.
"""

import torch
from tqdm import tqdm
from itertools import product
from argparse import ArgumentParser

try:
    from ..model.GeneticAlgorithm import GeneticAlgorithm
    from ..model.Population import Population

except ImportError:
    from os.path import join
    from sys import path

    path.insert(1, join(path[0], ".."))

    from model.GeneticAlgorithm import GeneticAlgorithm
    from model.Population import Population


def generate_fitness_map(gene_bound: int) -> torch.tensor:
    """Generates a random mapping to all possible gene values.
    args:
        gene_bound (int): Upper bound of each gene.
        gene_length (int): Length of each gene.
    """

    return torch.randint(gene_bound + 1, (gene_bound,))


def fitness_function(
    population: torch.tensor, fitness_map: torch.tensor
) -> torch.tensor:
    """Returns the sum of the fitness map for each individual's genes."""

    fitness = torch.zeros(population.shape[0])

    for index, individual in enumerate(population):
        fitness[index] = sum(fitness_map[gene] for gene in individual)

    return fitness


def decode(
    population: torch.tensor, fitness_map: torch.tensor
) -> torch.tensor:
    """Convert indices to values."""

    indices = product(range(len(population)), range(len(population[0])))

    decoded_population = torch.zeros_like(population)

    for individual, gene in indices:
        gene_point = population[individual, gene]
        decoded_population[individual, gene] = fitness_map[gene_point]

    return decoded_population


def parse_args():
    """Parse command line arguments."""

    parser = ArgumentParser(
        description="Demonstrate the genetic algorithm using sum as fitness.",
        add_help=True,
    )

    # Algorithm arguments

    parser.add_argument(
        "--generations",
        type=int,
        default=10_000,
        help="Number of generations to run.",
    )

    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.1,
        help="Probability of mutation.",
    )

    parser.add_argument(
        "--crossover-probability",
        type=float,
        default=0.3,
        help="Probability of crossover.",
    )

    # Population arguments

    parser.add_argument(
        "--gene-size",
        type=int,
        default=10,
        help="Length of each gene.",
    )

    parser.add_argument(
        "--gene-bound",
        type=int,
        default=1000,
        help="Upper bound of each gene.",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Size of the population.",
    )

    # Physical Implementations

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the fitness history.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    torch.random.manual_seed(args.seed)

    print("----------------------")
    print("Genetic Algorithm Demo")
    print("----------------------")

    # Generates a random map
    fitness_map = generate_fitness_map(gene_bound=args.gene_bound)

    # Initialize population and solver

    population = Population(
        size=args.population_size,
        gene_size=args.gene_size,
        gene_bound=args.gene_bound,
        device=args.device,
        fitness_function=lambda x: fitness_function(x, fitness_map),
    )

    solver = GeneticAlgorithm(
        population=population,
        mutation_probability=args.mutation_probability,
        crossover_probability=args.crossover_probability,
    )

    print(f"Initial fitness: {solver.history.best_fitness:.2f}")
    print(f"Initial population:\n{solver.history.best_population}")

    # Run the algorithm

    print("----------------------")

    for generation in (progress := tqdm(range(args.generations))):
        solver.evolve()
        progress.set_description(
            f"Generation {generation}: {solver.history.best_fitness:.2f}"
        )

    print("----------------------")

    # Print results

    decoded_population = decode(solver.history.best_population, fitness_map)

    print(f"Best fitness: {solver.history.best_fitness:.2f}")
    print(f"Best population:\n{solver.history.best_population}")
    print(f"Best population (decoded):\n{decoded_population}")

    # Compare to optimal solution

    print("----------------------")

    optimal = args.gene_size * args.gene_bound
    print(f"Ratio to optimal: {solver.history.best_fitness/optimal}")

    print("----------------------")

    if args.plot:
        solver.history.plot()


if __name__ == "__main__":
    main()
