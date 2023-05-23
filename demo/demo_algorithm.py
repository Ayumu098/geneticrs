"""
Demonstrate the general algorithm using sum as the fitness function.
"""

import torch
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

    # Print algorithm arguments
    print("Settings")

    print("----------------------")
    print(f"Seed:            {args.seed}")
    print(f"Generations:     {args.generations}")
    print("----------------------")

    print(f"Mutation Rate:   {args.mutation_probability}")
    print(f"Crossover Rate:  {args.crossover_probability}")
    print("----------------------")

    print(f"Gene bound:      {args.gene_bound}")
    print(f"Gene size:       {args.gene_size}")
    print(f"Population size: {args.population_size}")
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

    print("Initial Population")
    print("----------------------")
    print(f"Best Fitness: {solver.history.best_fitness:.2f}")
    print(f"Population:\n{solver.history.best_population.numpy()}")

    # Run the algorithm

    print("----------------------")
    print("Running Algorithm")
    print("----------------------")

    for _ in (progress := tqdm(range(args.generations))):
        solver.evolve()
        progress.set_description(
            f"Best Fitness: {solver.history.best_fitness:.2f}",
        )

    print("----------------------")
    print("Best Population")
    print("----------------------")

    # Print results

    print(f"Best fitness: {solver.history.best_fitness:.2f}")
    print(f"Population:\n{solver.history.best_population.numpy()}")

    # Compare to optimal solution

    print("----------------------")
    print("Evaluation")
    print("----------------------")

    optimal = args.gene_size * args.gene_bound
    print(f"Optimal solution:   {optimal}")
    print(f"Generated solution: {int(solver.history.best_fitness)}")
    print(f"Ratio to optimal:   {solver.history.best_fitness/optimal}")

    print("----------------------")

    if args.plot:
        solver.history.plot()


if __name__ == "__main__":
    main()
