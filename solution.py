from model.GeneticAlgorithm import GeneticAlgorithm, Population
from crs.RegistrationSystem import RegistrationSystem
from data.ScheduleDataset import ScheduleDataset

from argparse import ArgumentParser
from tqdm import tqdm

import torch


def parse_args():
    """Parse command line arguments."""

    parser = ArgumentParser(
        description="Demonstrate the genetic algorithm using sum as fitness.",
        add_help=True,
    )

    # Genetic Algorithm arguments

    parser.add_argument(
        "--generations",
        type=int,
        default=1000,
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
        "--gene_size",
        type=int,
        default=20,
        help="Length of each gene.",
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

    # User settings

    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Size of the population.",
    )

    parser.add_argument(
        "--weight-probability",
        type=float,
        default=1.0,
        help="Weight of the probability sum in the fitness function.",
    )

    parser.add_argument(
        "--weight-balance",
        type=float,
        default=1.0,
        help="Weight of the probability balance in the fitness function.",
    )

    parser.add_argument(
        "--weight-overlap",
        type=float,
        default=1.0,
        help="Weight of the shcedule overlap in the fitness function.",
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

    print("-----------------------------------------")
    print("Genetic Algorithm Scheduling for UPD CRS")
    print("-----------------------------------------")

    # Load dataset
    dataset = ScheduleDataset("input.csv")

    # Initialize RegistrationSystem
    system = RegistrationSystem(dataset)

    # Initialize population and solver

    weights = torch.tensor(
        [args.weight_probability, args.weight_balance, args.weight_overlap]
    )

    population = Population(
        fitness_function=lambda x: system.fitness_function(x, weights),
        size=args.population_size,
        gene_size=args.gene_size,
        gene_bound=len(dataset) - 1,
        device=args.device,
    )

    solver = GeneticAlgorithm(
        population=population,
        mutation_probability=args.mutation_probability,
        crossover_probability=args.crossover_probability,
    )

    print(f"Initial fitness: {solver.history.best_fitness:.2f}")
    print(f"Initial population:\n{solver.history.best_population.numpy()}")

    # Run the algorithm

    print("-----------------------------------------")

    for generation in (progress := tqdm(range(args.generations))):
        solver.evolve()
        progress.set_description(
            f"Generation {generation}: {solver.history.best_fitness:.2f}"
        )

    print("-----------------------------------------")

    # Print results

    print(f"Best fitness: {solver.history.best_fitness:.2f}")
    print(f"Best population:\n{solver.history.best_population.numpy()}")

    # Assess using RegistrationSystem

    print("-----------------------------------------")
    print("Assessing the best individual in the CRS")

    assessment_length = 1000
    enlistment_count = torch.zeros(assessment_length)
    best_individual = solver.history.best_population[0]

    for assessment in tqdm(range(assessment_length)):
        enlistment_count[assessment] = len(system(best_individual))

    print("-----------------------------------------")

    print(f"Average enlistment count: {enlistment_count.mean()}")

    print("-----------------------------------------")

    if args.plot:
        solver.history.plot()


if __name__ == "__main__":
    main()
