!pip install --upgrade pandas 

#conda update pandas

import pandas as pd
print(pd.__version__)

!pip show pandas
__________________________________________________________________________________
import pandas as pd

# https://github.com/manishjainstorage/Advanced-Python/blob/main/StockMarketData-13krows.xlsx
# Define the correct raw URL of the .xlsx file
url = "https://github.com/manishjainstorage/Advanced-Python/raw/main/StockMarketData-13krows.xlsx"

# Read the .xlsx file into a DataFrame, specifying the engine
df = pd.read_excel(url, engine='openpyxl')

# Display the first few rows of the DataFrame
df.head()
__________________________________________________________________________________

import pandas as pd
import polars as pl
import time

# Load large dataset using Pandas
start_time = time.time()
df_pandas = pd.read_excel("https://github.com/manishjainstorage/Advanced-Python/raw/main/StockMarketData-13krows.xlsx")
print(f'Pandas load time: {time.time() - start_time} seconds')

# Load large dataset using Polars
start_time = time.time()
df_polars = pl.read_excel("https://github.com/manishjainstorage/Advanced-Python/raw/main/StockMarketData-13krows.xlsx")
print(f'Polars load time: {time.time() - start_time} seconds')
___________________________________________________________________________________

def a(n):
  i = 1
  while i<=n:
    yield i
    i+=1

gen = a(5)

print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))
________________________________________________________________________

import pandas as pd
import polars as pl
import numpy as np
import time

# Define the size of the dataset
N = 10**7  # 10 million rows

# Generate a large dataset
data = {
    'a': np.random.rand(N),
    'b': np.random.rand(N)
}

# Function to perform a computation in Pandas
def pandas_computation(data):
    df = pd.DataFrame(data)
    # Perform a computation (e.g., summing the product of two columns)
    result = (df['a'] * df['b']).sum()
    return result

# Function to perform a computation in Polars
def polars_computation(data):
    df = pl.DataFrame(data)
    # Perform a computation (e.g., summing the product of two columns)
    result = (df['a'] * df['b']).sum()
    return result

# Measure execution time for Pandas
start_time = time.time()
pandas_result = pandas_computation(data)
pandas_time = time.time() - start_time
print(f"Pandas result: {pandas_result}, Time taken: {pandas_time:.2f} seconds")

# Measure execution time for Polars
start_time = time.time()
polars_result = polars_computation(data)
polars_time = time.time() - start_time
print(f"Polars result: {polars_result}, Time taken: {polars_time:.2f} seconds")

________________________________________________________________________________________


df = pl.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [7, 8, 9]
})

# Apply a chain of expressions
df = df.with_columns([
    (pl.col("a") * pl.col("b")).alias("a_times_b"),
    (pl.col("c") + 5).alias("c_plus_5")
])

df

____________________________________________________________________________________

df = pl.DataFrame({
    "time": [1, 2, 3, 4, 5],
    "value": [10, 20, 30, 40, 50]
})

# Dynamic window function with a window size of 2
df = df.with_columns([
    pl.col("value").rolling_sum(window_size=3).alias("rolling_sum")
])

df

_______________________________________________________________________________


""" NumPy Array Operations

NumPy is the fundamental package for numerical computing in Python.
It provides powerful array objects and functions that support vectorization """

import numpy as np

# Create two large NumPy arrays
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Element-wise addition (vectorized operation)
c = a + b  # This operation is executed in C, making it faster than a Python loop
print(c)

_______________________________________________________________________________


"""Broadcasting

Broadcasting allows NumPy to perform arithmetic operations on arrays of 
different shapes without the need for explicit loops."""

import numpy as np

# Create a 1D array and a 2D array
a = np.array([1, 2, 3,4])
b = np.array([[10], [20], [30]])

# Broadcasting the 1D array to match the shape of the 2D array
result = a + b
print(result)

________________________________________________________________

"""Using Numba

Numba is a just-in-time (JIT) compiler that translates a subset of Python and NumPy 
code into fast machine code. 

It can significantly speed up array operations."""

import numpy as np
from numba import jit

@jit(nopython=True)
def add_arrays(a, b):
    return a + b

# Create large arrays
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Call the JIT-compiled function
c = add_arrays(a, b)
print(c)

_________________________________________________________________________


"""SciPy for Scientific Computing

SciPy builds on NumPy and provides additional functionality for optimization,
 integration, interpolation, 
and other scientific computations, often with vectorized implementations."""

from scipy import integrate
import numpy as np

# Define a function
def f(x):
    return x**2

# Vectorized integration
result, error = integrate.quad(f, 0, 1)  # Integrates f(x) from 0 to 1
print(result, error)

_________________________________________________________________

""" Utilize Vectorization

Description: Replace explicit loops with NumPy's built-in array operations, 
which are optimized and run in compiled C code"""

import numpy as np

# Instead of this:
a = np.random.rand(1000000)
b = np.random.rand(1000000)
c = np.empty_like(a)
for i in range(len(a)):
    c[i] = a[i] + b[i]

# Use vectorized operations:
c = a + b  # Faster and more concise

_________________________________________________________________________

"""Pre-allocate Arrays

Description: When creating arrays, pre-allocate them with the desired shape 
and size to avoid resizing during operations. """

# Instead of dynamically growing an array:
result = []
for i in range(1000):
    result.append(i**2)
result = np.array(result)  # Conversion to NumPy array afterward

# Pre-allocate:
result = np.empty(1000)
for i in range(1000):
    result[i] = i**2

____________________________________________________________________________


"""Avoid Unnecessary Copies

Description: Be mindful of operations that create copies of arrays 
(e.g., slicing or reshaping). Use views when possible."""

# Slicing creates a view, while some operations create copies
a = np.array([1, 2, 3, 4, 5])
b = a[1:3]  # b is a view, changes in a reflect in b

# Use np.copy() if you need an independent copy
c = np.copy(a[1:3])  # Creates a new array

____________________________________________________________________________

""" Use In-place Operations

Description: Modify existing arrays in place to save memory and improve performance."""

a = np.array([1, 2, 3])
a += 1  # In-place addition, modifies a directly

______________________________________________________________________________

"""Profile Your Code

Description: Use profiling tools like cProfile or line_profiler to identify bottlenecks
 in your code and optimize those specific areas."""

import cProfile

def compute():
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)
    return a + b

cProfile.run('compute()')

_____________________________________________________________________

#!pip install pandas-profiling

import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv("StockMarketData-13krows_csv.csv")

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("output.html")

________________________________________________________

"""
b. Using Line Profiler

line_profiler is a third-party module that provides line-by-line profiling of functions.

"""

# Step 1: Install Line Profiler
!pip install line_profiler

# Step 2: Load the Line Profiler Extension
%load_ext line_profiler

# Step 3: Define the function to profile
def my_function():
    total = 0
    for i in range(10000):
        total += i ** 2
    return total

# Run the function first (optional)
my_function()

# Step 4: Use lprun to profile the function
%lprun -f my_function my_function()


_________________________________________________________________________________________


""" 3. Code Optimization techniques

Once you identify bottlenecks, consider the following optimization strategies:

a. Optimize Algorithms

Choose Efficient Algorithms: Evaluate whether a more efficient algorithm can be used.

Algorithm Complexity: Analyze the time complexity of your algorithms and look for improvements.
b. Utilize Vectorization

Replace explicit loops with vectorized operations using libraries like NumPy."""

import numpy as np
import time

# Original algorithm: O(n^2) complexity
def sum_of_squares_nested(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i**2
    return total

# Optimized algorithm: O(n) complexity
def sum_of_squares_optimized(n):
    return n * (n - 1) * (2 * n - 1) // 6  # Formula for sum of squares

# Vectorized operation using NumPy
def sum_of_squares_vectorized(n):
    arr = np.arange(n)
    return np.sum(arr**2)

# Testing the algorithms
n = 10000

# Measure time for original algorithm
start_time = time.time()
result_nested = sum_of_squares_nested(n)
end_time = time.time()
print(f"Nested Sum of Squares Result: {result_nested}, Time: {end_time - start_time:.5f} seconds")

# Measure time for optimized algorithm
start_time = time.time()
result_optimized = sum_of_squares_optimized(n)
end_time = time.time()
print(f"Optimized Sum of Squares Result: {result_optimized}, Time: {end_time - start_time:.5f} seconds")

# Measure time for vectorized algorithm
start_time = time.time()
result_vectorized = sum_of_squares_vectorized(n)
end_time = time.time()
print(f"Vectorized Sum of Squares Result: {result_vectorized}, Time: {end_time - start_time:.5f} seconds")
____________________________________________________________________________________________
