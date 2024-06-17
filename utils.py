import random

def initialize_population(pop_size, genome_length):
    return [[random.uniform(-1, 1) for _ in range(genome_length)] for _ in range(pop_size)]

def evaluate_population(population, fitness_function):
    return [fitness_function(individual) for individual in population]

if __name__ == "__main__":
    def sample_fitness_function(individual):
        return sum(individual)

    population = initialize_population(10, 5)
    fitness_scores = evaluate_population(population, sample_fitness_function)

    print("Initial Population:")
    for individual in population:
        print(individual)

    print("\nFitness Scores:")
    print(fitness_scores)
