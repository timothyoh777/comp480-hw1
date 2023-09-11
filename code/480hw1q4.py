import math
import random
import string
import sys
import pandas as pd
import matplotlib.pyplot as plt


from prettytable import PrettyTable
from bitarray import bitarray
from sklearn.utils import murmurhash3_32

"""
4.4.1
"""

# m is the desired hash table size
def hash_function_factory(m, seed=0):
    def hash_function(key):
        return murmurhash3_32(key, seed=seed, positive=True) % m
    return hash_function

"""
4.1.2

In class we saw the basic Bloom filter, where we used k independent random hash-functions h1, h2, ..., hk
to hash a set S of N elements into an array A of R bits. Recall that the formulas for computing the best k
to use for a given key set size M and maximum false positive rate at given range,

k = (R/N)ln(2)

with false positive rate (f)

f = 0.618^(R/N)

therefore, to the necessary values to init our bloom filter for a given certain f and N, 

k = log_(0.618)(f) * ln(2)
R = log_(0.618)(f) * N
N = R / log_(0.618)(f) (given)

then, we want k independent hash function that hash to a bit array of size R for N items
"""

# We want our bit array ranges to be a power of two. therefore, given a calculated range R, we want to round up to the nearest power of two
# greater than or equal to R
def round_up_to_power_of_two(num):
    if num <= 0:
        return 1
    # Check if num is already a power of two
    if (num & (num - 1)) == 0:
        return num
    # Find the position of the most significant bit (MSB) in num
    msb_position = 0
    while num > 0:
        num >>= 1
        msb_position += 1
    # Set all lower bits to 1 to create the next power of two
    rounded_num = 1 << msb_position
    return rounded_num

class BloomFilter:
    def __init__(self, n, fp_rate=None, r=None):
        # number of expected items to store
        self.n = n
        # calculate k 
        self.k = (0.7 * r) // self.n if r else int(math.log(fp_rate, 0.618) * math.log(2))
        # calculate bitarray size
        self.r = r if r else round_up_to_power_of_two(int(math.log(fp_rate, 0.618) * self.n))

        self.hash_array = bitarray(self.r)
        self.hash_array.setall(0)

        # initialize our k independent hash function that map to a bit array of size self.r
        # in order to make this experiement reproducible, for k different independent hash functions we 
        # iterate k times from 0 => k * 5 in increments of 5 to have constant but different murmur hash seeds
        self.hash_functions = []
        for seed in range(0, int(self.k) * 5, 5):
            generated_function = hash_function_factory(self.r, int(seed))
            self.hash_functions.append(generated_function)

    def insert(self, key) -> None:
        for hash_function in self.hash_functions:
            generated_hash = hash_function(key)
            self.hash_array[generated_hash] = 1

    def test(self, key) -> bool:
        for hash_function in self.hash_functions:
            test_hash = hash_function(key)
            if self.hash_array[test_hash]:
                continue
            return False
        return True
    

"""
4.1.3
"""

# Set a fixed random seed for consistency
random.seed(2938475849302384754839)
# Generate the membership set with 10,000 unique integers
membership_set = set()
while len(membership_set) < 10000:
    membership_set.add(random.randint(10000, 99999))

# Generate the test set with 1000 unique integers not in the membership set
test_set_not_in_membership = set()
while len(test_set_not_in_membership) < 1000:
    num = random.randint(10000, 99999)
    if num not in membership_set:
        test_set_not_in_membership.add(num)

# Generate the test set with 1000 integers randomly selected from the membership set
test_set_from_membership = random.sample(list(membership_set), 1000)

# Convert the sets to lists 
membership_list = list(membership_set)
test_list_from_membership = test_set_from_membership
test_list_not_in_membership = list(test_set_not_in_membership)

def evaluate_bloom_filter(membership_list, test_list_not_in_membership, test_list_from_membership, bloom_filter):
    for key in membership_list:
        bloom_filter.insert(key)

    false_positives = 0
    for key in test_list_not_in_membership:
        if bloom_filter.test(key):
            false_positives += 1

    false_negatives = 0
    for key in test_list_from_membership:
        if not bloom_filter.test(key):
            false_negatives += 1

    return false_positives, false_negatives

def calculate_warmup_false_positives():
    false_positive_rates = [0.01, 0.001, 0.0001]

    for rate in false_positive_rates:
        bloom_filter = BloomFilter(fp_rate=rate, n=10000)
        false_positives, false_negatives = evaluate_bloom_filter(membership_list, test_list_not_in_membership, test_list_from_membership, bloom_filter)
        
        print(f"{rate} FALSE POSITIVE RATE")
        print(f"False positives: {false_positives}")
        print(f"False positive rate: {false_positives / 1000}")
        print(f"False negatives: {false_negatives}")
        print(f"False negative rate: {false_negatives / 1000}")

"""
4.2

Sample 1000 urls from urllist as the test set and 1000 random strings as false urls. Report the false
positive rate and memory usage of your design of bloom filter. Plot the false positive rate with
memory by varying R. (use k=0.7R/N, N=377871 in this dataset).
"""

data = pd.read_csv("user-ct-test-collection-01.txt", sep="\t")
urllist = data.ClickURL.dropna().unique()

# helper function to generate random strings
def random_string(length):
    # Define the characters from which to generate the string
    characters = string.ascii_letters + string.digits  # You can customize this further
    
    # Use random.choices to generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))
    
    return random_string


url_in_urllist = random.sample(list(urllist), 100)
url_not_in_urllist = [random_string(random.randint(10, 50)) for _ in range(1000)]

# Create a map to hold onto false positive counts per K (bit array size)
false_positive_rates = {}

# Create arrays to hold onto memory sizes
bit_array_sizes = []  # Store bit array sizes
bloom_filter_memory = []  # Store Bloom filter memory usage
python_hashset_memory = []  # Store Python hash set memory usage

# We iterate until 2^24 size hash table => 2**23 is size of bit array for 377871 insertions to
# have a false positive rate of 0.001 
bit_array_size = 2
while bit_array_size <= 2**24:
    bloom_filter = BloomFilter(n=377871, r=bit_array_size)
    python_hashset = set()

    for url in urllist:
        bloom_filter.insert(url)
        python_hashset.add(url)


    false_positives, false_negatives = evaluate_bloom_filter(urllist, url_not_in_urllist, url_in_urllist, bloom_filter)
    false_positive_rates[bit_array_size] = false_positives / 1000

    bloom_filter_memory_usage = sys.getsizeof(bloom_filter)
    python_memory_usage = sys.getsizeof(python_hashset)

    bit_array_sizes.append(bit_array_size)
    bloom_filter_memory.append(bloom_filter_memory_usage)
    python_hashset_memory.append(python_memory_usage)

    bit_array_size *= 2

def show_false_positive_per_bit_array():
    # Extract keys (bit array sizes) and values (false positive rates)
    sizes = list(false_positive_rates.keys())
    rates = list(false_positive_rates.values())

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, rates, marker='o', linestyle='-')
    plt.title('False Positive Rates vs. Bit Array Size (K)')
    plt.xlabel('Bit Array Size (K)')
    plt.ylabel('False Positive Rate')
    plt.grid(True)
    # Show the plot
    plt.show()

def show_memory_usage():
    # Create a table
    table = PrettyTable()
    table.field_names = ["Bit Array Sizes", "Bloom Filter Memory (bytes)", "Python HashSet Memory (bytes)"]

    # Add data to the table
    for size, bloom, python in zip(bit_array_sizes, bloom_filter_memory, python_hashset_memory):
        table.add_row([size, bloom, python])

    # Display the table
    print(table)



# driver code
if __name__ == "__main__":
   calculate_warmup_false_positives()
   print("\n")
   show_false_positive_per_bit_array()
   show_memory_usage()
