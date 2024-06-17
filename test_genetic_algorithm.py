import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from selection import tournament_selection, roulette_wheel_selection
from crossover import single_point_crossover, two_point_crossover
from mutation import mutate
from utils import initialize_population, evaluate_population
from config import tickers

def load_formatted_data(ticker):
    data_dir = os.path.join(os.path.dirname(__file__), 'DATA')
    file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_close_prices.csv")
    return prepare_lstm_data(file_path)

def prepare_lstm_data(file_path, look_back=60):
    data_df = pd.read_csv(file_path)
    data = data_df.filter(['close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i-look_back:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler

def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@tf.function(reduce_retracing=True)
def predict_with_model(model, x_test):
    return model(x_test, training=False)

def lstm_fitness_function(individual, x_train, y_train, x_test, y_test):
    units = int(np.clip(individual[0] * 100, 1, 200))  # Ensure units are between 1 and 200
    dropout_rate = np.clip(1 / (1 + np.exp(-individual[1])), 0.1, 0.5)  # Ensure dropout_rate is between 0.1 and 0.5
    epochs = int(np.clip(individual[2] * 100, 1, 100))  # Ensure epochs are between 1 and 100

    model = build_lstm_model((x_train.shape[1], 1), units=units, dropout_rate=dropout_rate)
    
    # Ensure that the data shapes are correct before training
    if x_train.shape[0] == 0 or y_train.shape[0] == 0 or x_test.shape[0] == 0 or y_test.shape[0] == 0:
        return float('-inf')
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    predictions = predict_with_model(model, x_test)
    mse = mean_squared_error(y_test, predictions)
    
    return -mse  # We want to minimize MSE, so return its negative as fitness

def genetic_algorithm(fitness_function, x_train, y_train, x_test, y_test, pop_size=100, genome_length=3, generations=100, mutation_rate=0.01, selection_method='tournament', crossover_method='single_point'):
    population = initialize_population(pop_size, genome_length)
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitness_scores = evaluate_population(population, lambda ind: fitness_function(ind, x_train, y_train, x_test, y_test))
        
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

        best_individual_gen, best_fitness_gen = max(zip(population, fitness_scores), key=lambda x: x[1])

        if best_fitness_gen > best_fitness:
            best_individual = best_individual_gen
            best_fitness = best_fitness_gen

        print(f"Generation {generation + 1}/{generations}: Best Fitness = {best_fitness}")

    return best_individual, best_fitness

if __name__ == "__main__":
    ticker = tickers[0]  # Use the first ticker for this example
    x_train, y_train, scaler = load_formatted_data(ticker)
    
    # Split the data into training and testing sets
    training_data_len = int(np.ceil(len(y_train) * .95))
    x_test = x_train[training_data_len:]
    y_test = y_train[training_data_len:]
    x_train = x_train[:training_data_len]
    y_train = y_train[:training_data_len]

    best_ind, best_fit = genetic_algorithm(
        fitness_function=lstm_fitness_function,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        pop_size=10,  # Adjust as needed
        genome_length=3,  # Units, Dropout Rate, Epochs
        generations=20,  # Adjust as needed
        mutation_rate=0.05,
        selection_method='tournament',
        crossover_method='single_point'
    )

    print("\nBest Individual:")
    print(best_ind)
    print("Best Fitness:")
    print(best_fit)
