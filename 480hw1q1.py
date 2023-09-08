import random
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import murmurhash3_32

P = 1048573
TEN_BIT_RANGE = 1024
NUM_KEYS = 5000
NUM_INPUT_BITS = 31
NUM_OUTPUT_BITS = 10
MURMUR_HASH_SEED = 1264528
# [a, b, c, d]
HASH_FUNCTION_PARAMS = [79898, 818327, 149554, 269194]

# Function 1: ((ax + b) mod P) mod 1024 (2-universal)
def hash_function_1(x, a, b):
    return ((a * x + b) % P) % TEN_BIT_RANGE

# Function 2: ((ax^2 + bx + c) mod P) mod 1024 (3-universal)
def hash_function_2(x, a, b, c):
    return ((a * x**2 + b * x + c) % P) % TEN_BIT_RANGE

# Function 3: ((ax^3 + bx^2 + cx + d) mod P) mod 1024 (4-universal)
def hash_function_3(x, a, b, c, d):
    return ((a * x**3 + b * x**2 + c * x + d) % P) % TEN_BIT_RANGE

# Function 4: Murmur hash
def murmur_hash(data):
    return murmurhash3_32(data, seed=MURMUR_HASH_SEED, positive=True) % TEN_BIT_RANGE

def generate_random_keys():
    return [random.randint(1, 2 ** NUM_INPUT_BITS - 1) for _ in range(NUM_KEYS)]

def flip_bit(key, bit_position):
    return key ^ (1 << bit_position)

def analyze_hash_function(hash_function, parameters, keys):
    output_bit_changes = {i: [0] * NUM_INPUT_BITS for i in range(10)}

    for key in keys:
        if hash_function == hash_function_1:
            original_hash = hash_function(key, *parameters[:2])  # Pass only 'a' and 'b'
        elif hash_function == hash_function_2:
            original_hash = hash_function(key, *parameters[:3])  # Pass 'a', 'b', and 'c'
        elif hash_function == hash_function_3:
            original_hash = hash_function(key, *parameters)      # Pass all parameters
        elif hash_function == murmur_hash:
            original_hash = hash_function(key)
    

        for input_bit in range(NUM_INPUT_BITS):
            flipped_key = flip_bit(key, input_bit)
            
            if hash_function == hash_function_1:
                new_hash = hash_function(flipped_key, *parameters[:2])  # Pass only 'a' and 'b'
            elif hash_function == hash_function_2:
                new_hash = hash_function(flipped_key, *parameters[:3])  # Pass 'a', 'b', and 'c'
            elif hash_function == hash_function_3:
                new_hash = hash_function(flipped_key, *parameters)      # Pass all parameters
            elif hash_function == murmur_hash:
                new_hash = hash_function(flipped_key)
                
            xor_result = original_hash ^ new_hash

            # For each bit position in the output, check if it has been flipped
            for output_bit in range(10):
                if (xor_result & (1 << output_bit)) != 0:
                    output_bit_changes[output_bit][input_bit] += 1

    probability_matrix = np.zeros((NUM_OUTPUT_BITS, NUM_INPUT_BITS))

    for i in range(10):
        for j in range(NUM_INPUT_BITS):
            probability_matrix[i][j] = output_bit_changes[i][j] / len(keys)

    return probability_matrix

def plot_heatmap(probability_matrix, function_name):
    # Determine the minimum and maximum probabilities across all hash functions
    lightest_probability = 0
    darkest_probability = 0.5

    # Set a consistent color range for all plots
    plt.imshow(
        probability_matrix, 
        cmap='Reds', 
        extent=[0, NUM_INPUT_BITS, 0, 10], 
        origin='lower', 
        aspect='auto', 
        vmin=lightest_probability,  # Set the minimum color value
        vmax=darkest_probability  # Set the maximum color value
    )
    
    plt.colorbar(label='Probability')
    plt.xlabel('Input Bit j')
    plt.ylabel('Output Bit i')
    plt.title(f"{function_name}: Probability that bit i will change if bit j is flipped")
    plt.show()


def main():
    parameters = HASH_FUNCTION_PARAMS
    print("Parameters for Hash Function:", parameters)

    keys = generate_random_keys()

    hash_functions = [hash_function_1, hash_function_2, hash_function_3, murmur_hash]

    for hash_function in hash_functions:
        start_time = time.time()
        probability_matrix = analyze_hash_function(hash_function, parameters, keys)
        print(f"Elapsed time for {hash_function.__name__}: {time.time() - start_time:.2f} seconds")
        plot_heatmap(probability_matrix, hash_function.__name__)

if __name__ == "__main__":
    main()
