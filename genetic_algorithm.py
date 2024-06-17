# Questo modulo mette tutto insieme e esegue l'algoritmo genetico.

import random
from selection import tournament_selection, roulette_wheel_selection
from crossover import single_point_crossover, two_point_crossover
from mutation import mutate
from utils import initialize_population, evaluate_population

def genetic_algorithm(fitness_function, pop_size=100, genome_length=10, generations=100, mutation_rate=0.01, selection_method='tournament', crossover_method='single_point'):
    population = initialize_population(pop_size, genome_length)
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitness_scores = evaluate_population(population, fitness_function)
        
        if selection_method == 'tournament':
            selected_population = tournament_selection(population, fitness_scores)
        elif selection_method == 'roulette':
            selected_population = roulette_wheel_selection(population, fitness_scores)
        else:
            raise ValueError("Invalid selection method")

        next_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[min(i+1, len(selected_population)-1)]
            
            if crossover_method == 'single_point':
                child1, child2 = single_point_crossover(parent1, parent2)
            elif crossover_method == 'two_point':
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                raise ValueError("Invalid crossover method")

            next_population.append(mutate(child1, mutation_rate))
            next_population.append(mutate(child2, mutation_rate))

        population = next_population[:pop_size]

        best_individual_gen = max(zip(population, fitness_scores), key=lambda x: x[1])[0]
        best_fitness_gen = max(fitness_scores)

        if best_fitness_gen > best_fitness:
            best_individual = best_individual_gen
            best_fitness = best_fitness_gen

        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_individual, best_fitness
