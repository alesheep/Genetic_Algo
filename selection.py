import random

def tournament_selection(population, fitness_scores, k=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), k)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    relative_fitness = [f / total_fitness for f in fitness_scores]
    probabilities = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
    
    selected = []
    for _ in range(len(population)):
        r = random.random()
        for (i, individual) in enumerate(population):
            if r <= probabilities[i]:
                selected.append(individual)
                break
    return selected

if __name__ == "__main__":
    population = [[random.uniform(-1, 1) for _ in range(5)] for _ in range(10)]
    fitness_scores = [random.random() for _ in range(10)]

    selected_tournament = tournament_selection(population, fitness_scores)
    selected_roulette = roulette_wheel_selection(population, fitness_scores)

    print("Tournament Selection Result:")
    for individual in selected_tournament:
        print(individual)

    print("\nRoulette Wheel Selection Result:")
    for individual in selected_roulette:
        print(individual)
