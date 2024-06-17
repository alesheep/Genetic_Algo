from genetic_algorithm import run_genetic_algorithm
from data_processing import load_data

if __name__ == "__main__":
    # Carica e prepara i dati
    data = load_data("path/to/stock_data.csv")

    # Esegui l'algoritmo genetico
    best_solution = run_genetic_algorithm(data)

    print("Best Solution:", best_solution)