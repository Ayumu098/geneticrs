from collections import defaultdict
from itertools import combinations
from functools import lru_cache
import torch


class RegistrationSystem:
    """Class that represents the University of the Philippines
    Computer Registration System that enlists subjects by lottery.
    """

    @lru_cache(maxsize=None)
    @staticmethod
    def overlap(schedule1: torch.tensor, schedule2: torch.tensor) -> bool:
        """Checks if two schedules overlap."""
        return torch.any(schedule1 + schedule2 >= 2)

    @staticmethod
    def enlist(probability: float):
        """Returns `True` if probability allowed for enlistment;
        Else, `False`.
        """
        assert 0 <= probability <= 1
        return torch.bernoulli(torch.tensor(probability))

    def __init__(self, dataset) -> None:
        self.names = []
        self.types = []
        self.schedules = []
        self.probabilities = []

        for class_name, class_type, probability, schedules in dataset:
            self.names.append(class_name)
            self.types.append(class_type)
            self.schedules.append(schedules)
            self.probabilities.append(probability)

    def __len__(self):
        return len(self.names)

    def fitness_score(
        self,
        individual: torch.tensor,
        weights: torch.tensor,
    ) -> torch.tensor:
        """
        Calculate the fitness score of a given individual
        wherein each element in the individual represents
        the index mapping to a subject in the ScheduleDataset.
        The basis of the fitness score is on the following:
            - The sum of the probabilities of the indexed subjects
            (the higher the better)
            - The standard deviation of the probabilities of the
            indexed subjects (the lower the better)
            - The overlap of the schedules of the indexed subjects
            (the lower the better)

        Args:
            individual (torch.tensor): A tensor of points.
            Each point represents or index a subject.
            weights (torch.tensor): A tensor of weights
            for the given bases of the fitness function.

        Returns:
            float: Fitness score of a given individual
        """

        # Remove redundant subjects
        individual = torch.unique(individual)

        # Organize gene points by their corresponding class type
        probabilities = defaultdict(lambda: torch.tensor(0.0))
        schedules = defaultdict(lambda: torch.zeros_like(self.schedules[0]))

        for gene in individual:
            probabilities[self.types[gene]] += self.probabilities[gene]
            schedules[self.types[gene]] += self.schedules[gene]
            schedules[self.types[gene]] = torch.clamp(
                schedules[self.types[gene]], max=1
            )

        probability_values = torch.tensor(list(probabilities.values()))

        # With schedules having higher probabilities
        # the overall enlistment chances increases
        probability_sum = probability_values.sum()
        probability_sum /= individual.shape[0]

        # Measures the balancing of the probabilities
        # A higher standard deviation means that some probabilities
        # are more prioritized than others. This is may be a problem
        # if it overprioritizes a subject that has a very low probability
        probability_standard_deviation = probability_values.std(unbiased=False)
        probability_standard_deviation /= individual.shape[0]

        # Counts the schedules that overlap with other schedules
        # except itself. Similar subject types are already added together
        # and clamped so the only consideration left is the overlap
        # between different subject types.
        schedule_tensor = torch.stack(list(schedules.values()))
        pair_indices = list(combinations(range(len(schedules.keys())), 2))

        overlaps = sum(
            RegistrationSystem.overlap(
                schedule_tensor[index1],
                schedule_tensor[index2],
            )
            for index1, index2 in pair_indices
        )
        overlaps = overlaps / max(len(pair_indices), 1)

        # Actual Fitness Function

        return torch.tensor(
            [probability_sum, 1 - probability_standard_deviation, 1 - overlaps]
        ).dot(weights)

    def fitness_function(
        self,
        population: torch.tensor,
        weights: torch.tensor,
    ) -> torch.tensor:
        """
        Calculate the fitness score of a given population
        wherein each element in the population represents
        an individual.

        Args:
            population (torch.tensor): A tensor of individuals.
            Each individual represents or index a subject.
            weights (torch.tensor): A tensor of weights
            for the given bases of the fitness function.

        Returns:
            torch.tensor: Fitness score of a given population
        """

        fitness = torch.zeros((population.shape[0]))

        for index, individual in enumerate(population):
            fitness[index] = self.fitness_score(individual, weights)

        return fitness

    def __call__(self, individual: torch.tensor) -> torch.tensor:
        """Simulates an enlistment distribution with the
        individual provided as the indices of the
        enlisted subjects for the batch run.

        Args:
            individual (torch.tensor): A tensor of subject indices.

        Returns:
            torch.tensor: A tensor of the enlisted subjects.
        """

        enlisted_subjects = []

        # Holds gene values wherein the schedules overlap
        # with already enlisted subjects.
        removed_subjects = []

        # Organize by overlapping schedules
        for gene in individual:
            if gene in removed_subjects:
                continue

            # Probabilistic Enlistment based on demand and availability
            if self.enlist(self.probabilities[gene]):
                enlisted_subjects.append(gene)

                # Set genes to be removed if they overlap
                for check in individual:
                    if RegistrationSystem.overlap(
                        self.schedules[gene], self.schedules[check]
                    ):
                        removed_subjects.append(check)

        return torch.tensor(enlisted_subjects)
