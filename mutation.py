import random

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(-1, 1)  # assuming the individual's genes are float numbers
    return individual

if __name__ == "__main__":
    individual = [random.uniform(-1, 1) for _ in range(5)]
    mutation_rate = 0.1

    mutated_individual = mutate(individual, mutation_rate)

    print("Original Individual:")
    print(individual)

    print("\nMutated Individual:")
    print(mutated_individual)
