import random

def single_point_crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2

def two_point_crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return child1, child2

if __name__ == "__main__":
    parent1 = [random.uniform(-1, 1) for _ in range(5)]
    parent2 = [random.uniform(-1, 1) for _ in range(5)]

    child1_sp, child2_sp = single_point_crossover(parent1, parent2)
    child1_tp, child2_tp = two_point_crossover(parent1, parent2)

    print("Single Point Crossover Result:")
    print("Child 1:", child1_sp)
    print("Child 2:", child2_sp)

    print("\nTwo Point Crossover Result:")
    print("Child 1:", child1_tp)
    print("Child 2:", child2_tp)
