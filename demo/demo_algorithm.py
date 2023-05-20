"""
Demonstrate the general algorithm using sum as the fitness function.
"""

from tqdm import tqdm
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
        default=0.4,
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

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Genetic Algorithm Demo")
    print("----------------------")

    # Initialize population and solver

    population = Population(
        size=args.population_size,
        gene_size=args.gene_size,
        gene_bound=args.gene_bound,
        device=args.device,
    )

    solver = GeneticAlgorithm(
        population=population,
        mutation_probability=args.mutation_probability,
        crossover_probability=args.crossover_probability,
    )

    print(f"Initial fitness: {solver.history.best_fitness:.2f}")
    print(f"Initial population:\n{solver.history.best_population}")

    # Run the algorithm

    for generation in (progress := tqdm(range(args.generations))):
        solver.evolve()
        progress.set_description(
            f"Generation {generation}: {solver.history.best_fitness:.2f}"
        )

    # Print results

    print(f"Best fitness: {solver.history.best_fitness:.2f}")
    print(f"Best population:\n{solver.history.best_population}")

    # Compare to optimal solution

    optimal = args.gene_size * args.gene_bound
    print(f"Ratio to optimal: {solver.history.best_fitness/optimal}")


if __name__ == "__main__":
    main()
