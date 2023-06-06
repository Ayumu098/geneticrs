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

    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Don't use a seed if enabled.",
    )

    parser.add_argument(
        "--stable",
        action="store_false",
        help="Will use elitism to ensure fitness doesn't decrease over time.",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default="input/input.csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.no_seed:
        torch.random.manual_seed(args.seed)

    # Load dataset
    dataset = ScheduleDataset(args.input_file)

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
        stable=args.stable,
    )

    print("-------------------------------------------")
    print("Genetic Algorithm Scheduling for UPD CRS")
    print("-------------------------------------------")

    # Print algorithm arguments
    print("Settings")
    print("-------------------------------------------")
    print(f"Stable:          {args.stable}")
    print(f"Seed:            {args.seed if not args.no_seed else torch.seed()}")
    print(f"Generations:     {args.generations}")
    print("-------------------------------------------")
    print(f"Mutation Rate:   {args.mutation_probability}")
    print(f"Crossover Rate:  {args.crossover_probability}")
    print("-------------------------------------------")
    print(f"Gene bound:      {len(dataset) - 1}")
    print(f"Gene size:       {args.gene_size}")
    print(f"Population size: {args.population_size}")
    print("-------------------------------------------")
    print(f"Weight - Probability:         {args.weight_probability}")
    print(f"Weight - Probability Balance: {args.weight_balance}")
    print(f"Weight - Schedule Overlap:    {args.weight_overlap}")
    print("-------------------------------------------")

    # Initial Population

    print("Initial Population")
    print("-------------------------------------------")
    print(f"Best Fitness: {solver.history.best_fitness:.2f}")
    print(f"Population:\n{solver.history.best_population.numpy()}")
    print("-------------------------------------------")

    # Assess using Registration System

    print("Assessing the initial population in the CRS")
    print("-------------------------------------------")

    assessment_length = 1000
    enlistment_counts = []

    for index, individual in enumerate(solver.history.best_population):
        enlistment_count = torch.zeros(assessment_length)

        for assessment in (progress := tqdm(range(assessment_length))):
            enlistments = len(system(individual))
            enlistment_count[assessment] = enlistments
            progress.set_description(f"Enlistment {index}: {enlistments}")
        enlistment_counts.append(enlistment_count)

    print("-------------------------------------------")

    for enlistment_count in enlistment_counts:
        print(
            f"Individual {index} Average # Enlisted Subjects:\
            {enlistment_count.mean():.5f}"
        )

    # Run the algorithm
    print("Running Algorithm")
    print("-------------------------------------------")

    for _ in (progress := tqdm(range(args.generations))):
        solver.evolve()
        progress.set_description(
            f"Best Fitness: {solver.history.best_fitness:.2f}",
        )
    print("-------------------------------------------")

    # Print results

    print("Best Population")
    print("-------------------------------------------")

    print(f"Best fitness: {solver.history.best_fitness:.2f}")
    print(f"Population:\n{solver.history.best_population.numpy()}")

    # Assess using Registration System

    print("-------------------------------------------")
    print("Assessing the best population in the CRS")
    print("-------------------------------------------")

    assessment_length = 1000
    enlistment_counts = []

    for index, individual in enumerate(solver.history.best_population):
        enlistment_count = torch.zeros(assessment_length)

        for assessment in (progress := tqdm(range(assessment_length))):
            enlistments = len(system(individual))
            enlistment_count[assessment] = enlistments
            progress.set_description(f"Enlistment {index}: {enlistments}")
        enlistment_counts.append(enlistment_count)

    print("-------------------------------------------")

    for index, enlistment_count in enumerate(enlistment_counts):
        print(
            f"Individual {index} Mean Enlisted Subject:\
            {enlistment_count.mean():.5f}"
        )

    print("-------------------------------------------")

    print("Suggested subjects to be enlisted:")

    for index1, enlistment in enumerate(solver.history.best_population):
        print("-------------------------------------------")
        print(f"Enlistment Option {index1}")
        print("-------------------------------------------")

        for index2, subject in enumerate(torch.unique(enlistment).sort()[0]):
            print(f"Subject {index2}: {system.names[subject]}")

    print("-------------------------------------------")

    if args.plot:
        solver.history.plot()


if __name__ == "__main__":
    main()
