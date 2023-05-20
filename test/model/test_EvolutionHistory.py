from ...model.GeneticAlgorithm import GeneticAlgorithm, Population, EvolutionHistory
import pytest


def test_initial_fitness() -> None:
    """Test initial conditions and initial fitness of history."""

    history = EvolutionHistory()

    # Test uninitialized history

    with pytest.raises(IndexError):
        history.initial_fitness

    with pytest.raises(IndexError):
        history.latest_fitness

    assert history.best_fitness is None

    # Test initialized history

    population = Population()
    solver = GeneticAlgorithm(population, history)
    initial_fitness = solver.population.fitness[0][0]

    assert history.best_fitness == solver.population.fitness[0][0]
    assert history.initial_fitness == solver.population.fitness[0][0]
    assert history.best_population.equal(solver.population.individuals)

    # Test history after evolution

    solver.evolve()

    assert history.initial_fitness == initial_fitness
    assert history.latest_fitness == solver.population.fitness[0][0]


def test_clear():
    """Test clearing history."""

    history = EvolutionHistory()
    population = Population()
    solver = GeneticAlgorithm(population, history)

    history.clear()

    # Test uninitialized history

    with pytest.raises(IndexError):
        history.initial_fitness

    with pytest.raises(IndexError):
        history.latest_fitness

    assert history.best_fitness is None

    # Test initialized history
    solver.evolve()
    solver.clear()

    # Test uninitialized history

    with pytest.raises(IndexError):
        history.initial_fitness

    with pytest.raises(IndexError):
        history.latest_fitness

    # Test if writing remains enabled
    solver.evolve()

    assert history.initial_fitness == population.fitness[0][0].item()
    assert history.latest_fitness == population.fitness[0][0].item()
    assert history.best_fitness == population.fitness[0][0].item()
    assert history.best_population.equal(population.individuals)


def test_freezing():
    """Test freezing and unfreezing history."""

    history = EvolutionHistory()
    history.freeze()

    # Test if writing is not applied

    population = Population()
    solver = GeneticAlgorithm(population, history)

    with pytest.raises(IndexError):
        history.initial_fitness

    with pytest.raises(IndexError):
        history.latest_fitness

    assert history.best_fitness is None

    solver.evolve()

    with pytest.raises(IndexError):
        history.initial_fitness

    with pytest.raises(IndexError):
        history.latest_fitness

    assert history.best_fitness is None

    # Test if writing is applied

    solver.history.unfreeze()
    solver.evolve()

    assert history.initial_fitness == population.fitness[0][0].item()
    assert history.latest_fitness == population.fitness[0][0].item()
    assert history.best_fitness == population.fitness[0][0].item()
    assert history.best_population.equal(population.individuals)

    # Test again if freeze is applied

    solver.history.freeze()
    solver.evolve(100)

    assert history.initial_fitness != population.fitness[0][0].item()
    assert history.latest_fitness != population.fitness[0][0].item()


def test_mutation():
    """Test mutation count is accurate."""

    history = EvolutionHistory()
    population = Population()
    solver = GeneticAlgorithm(
        population=population,
        history=history,
        mutation_probability=1.0,
        crossover_probability=0.0,
    )

    assert history.mutation[-1] == 0
    assert history.crossover[-1] == 0

    solver.evolve()

    assert history.mutation[-1] == population.size
    assert history.crossover[-1] == 0


def test_crossover():
    """Test mutation count is accurate."""

    history = EvolutionHistory()
    population = Population()
    solver = GeneticAlgorithm(
        population=population,
        history=history,
        mutation_probability=0.0,
        crossover_probability=1.0,
    )
    assert history.mutation[-1] == 0
    assert history.crossover[-1] == 0

    solver.evolve()

    assert history.mutation[-1] == 0
    assert history.crossover[-1] == population.size


def test_final_fitness():
    """Test if final fitness is accurate."""

    history = EvolutionHistory()
    population = Population()
    solver = GeneticAlgorithm(population, history)

    initial_fitness = population.fitness[0][0]

    assert history.latest_fitness == history.initial_fitness == initial_fitness

    # Test history after evolution

    solver.evolve(100)

    assert history.latest_fitness != initial_fitness
    assert history.latest_fitness != history.initial_fitness
    assert history.latest_fitness == population.fitness[0][0].item()


def test_best_fitness():
    """Test if best fitness is accurate."""

    # Initialization

    history = EvolutionHistory()
    population = Population(mutation_probability=0, crossover_probability=0)
    solver = GeneticAlgorithm(population, history)

    assert history.best_fitness == history.initial_fitness

    # Evolution

    solver.evolve(1000)

    assert history.best_fitness == max(history.fitness)
