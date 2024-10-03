import time
from functools import lru_cache

@lru_cache(maxsize=None)  # Cache results without a size limit
def expensive_function(n):
    """Simulates an expensive computation."""
    time.sleep(2)  # Simulate a delay
    return n * n

# Using the cached function
start_time = time.time()
print(expensive_function(4))  # First call, computes the value
print(f"Time taken: {time.time() - start_time} seconds")

start_time = time.time()
print(expensive_function(4))  # Second call, retrieves from cache
print(f"Time taken: {time.time() - start_time} seconds")

start_time = time.time()
print(expensive_function(4))  # Second call, retrieves from cache
print(f"Time taken: {time.time() - start_time} seconds")

start_time = time.time()
print(expensive_function(5))  # Second call, retrieves from cache
print(f"Time taken: {time.time() - start_time} seconds")

start_time = time.time()
print(expensive_function(5))  # Second call, retrieves from cache
print(f"Time taken: {time.time() - start_time} seconds")

_________________________________________________________________________________________

import time
from functools import lru_cache

# Simulated API or database call
@lru_cache(maxsize=5)  # Caches up to 5 recent calls
def fetch_data_from_db(query):
    print(f"Fetching data for query: {query}")
    time.sleep(2)  # Simulate a time-consuming database query
    return f"Results for {query}"

# Testing the caching
if __name__ == "__main__":
    print(fetch_data_from_db("SELECT * FROM users WHERE id = 1"))
    print(fetch_data_from_db("SELECT * FROM users WHERE id = 2"))
    # This call will be cached, so no delay
    print(fetch_data_from_db("SELECT * FROM users WHERE id = 1"))
___________________________________________________________________________________________________
def memoize(func):
    cache = {}

    def wrapper(n):
        if n not in cache:
            cache[n] = func(n)
        return cache[n]

    return wrapper

@memoize
def fibonacci(n):
    """Calculates the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Using the memoized Fibonacci function
print(fibonacci(10))  # Computes the value
print(fibonacci(10))  # Retrieves from cache
_________________________________________________________________________________________________

"""
Below is an example demonstrating how to cache results using joblib.Memory.
This will save the results of an expensive function call to disk,
 so it can be reused in subsequent calls with the same arguments.
"""

from joblib import Memory
import time

# Create a Memory object to store cache results
memory = Memory('./cachedir', verbose=0)

@memory.cache
def expensive_function(n):
    """Simulates an expensive computation."""
    time.sleep(2)  # Simulate a delay
    return n * n

# Using the cached function
start_time = time.time()
print(expensive_function(4))  # First call, computes the value
print(f"Time taken: {time.time() - start_time} seconds")

start_time = time.time()
print(expensive_function(4))  # Second call, retrieves from cache
print(f"Time taken: {time.time() - start_time} seconds")

start_time = time.time()
print(expensive_function(5))  # Computes a different value
print(f"Time taken: {time.time() - start_time} seconds")
________________________________________________________________________________

!pip install dask[complete]
import dask.array as da
import dask

# Create a large Dask array
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# Perform a computation (e.g., mean)
mean_result = x.mean()

# Trigger the computation
result = mean_result.compute()

print(f"The mean of the array is: {result}")
______________________________________________________

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

spark = SparkSession.builder.appName("example_spark").getOrCreate()

data = [(i,i*2) for i in range(10000000)]

columns = ["id" ,"value"]
df = spark.createDataFrame(data, columns)

start_time = time.time()
result = df.select(avg("value")).collect()[0][0]
end_time = time.time()

print(f"Spark Result: {result}, Time: {end_time - start_time:.5f} seconds")

____________________________________________________________________________________

import time
data = [(i,i*2) for i in range(10000000)]
start_time = time.time()
meanvaluenospark = sum([x[1] for x in data])/len(data)
end_time = time.time()
print(f"Spark Result: {meanvaluenospark}, Time: {end_time - start_time:.5f} seconds")

______________________________________________________________________________________
import dask.array as da
import dask

# Step 1: Create a large Dask array with chunking
# Create an array of random numbers (e.g., 10 million elements) and chunk it into smaller arrays
large_array = da.random.random(size=(10000000,), chunks=(1000000,))

# Step 2: Define an optimized algorithm function
def optimized_computation(x):
    """A simple optimized computation: Calculate the square and then the mean."""
    return da.mean(x ** 2)

# Step 3: Perform the computation
# Use Dask to apply the optimized function to the large array
mean_result = optimized_computation(large_array)

# Step 4: Trigger computation (this uses load balancing internally)
result = mean_result.compute()  # This will execute the task in parallel, leveraging multiple cores

# Step 5: Print the result
print(f"The mean of the squares of the array is: {result}")
_______________________________________________________________________________________________________

"""
Event Loop:

The asyncio library uses an event loop to schedule and run asynchronous tasks.
The event loop manages when and how the coroutines are executed,
providing concurrency without requiring multiple threads.

Tasks:

Coroutines can be wrapped into Task objects that run asynchronously.
Tasks allow the program to execute multiple coroutines simultaneously (in an interleaved manner).
"""

"""
Python program that demonstrates the use of the asyncio library and an event loop to schedule
and run asynchronous tasks. This example
will show how multiple coroutines can be executed concurrently without blocking the main thread:

"""
import asyncio

# Define an asynchronous function that simulates a task
async def task(name, delay):
    print(f"Task {name} started")  # Indicate the task has started
    await asyncio.sleep(delay)  # Simulate an I/O-bound operation (e.g., network request)
    print(f"Task {name} completed after {delay} seconds")  # Indicate the task has completed

# Define the main asynchronous function
async def main():
    # Create a list of tasks with varying delays
    tasks = [
        task("A", 2),  # Task A takes 2 seconds
        task("B", 1),  # Task B takes 1 second
        task("C", 3),  # Task C takes 3 seconds
    ]

    # Use asyncio.gather to run tasks concurrently
    await asyncio.gather(*tasks)

# Directly await the `main` function in Colab
await main()

_____________________________________________________________________________________________

#!pip install aiohttp beautifulsoup4 nest_asyncio

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# List of URLs to scrape
urls = [
    "https://example.com",
    "https://httpbin.org/get",
    "https://quotes.toscrape.com/",
    # Add more URLs as needed
]

async def fetch(session, url):
    async with session.get(url) as response:
        # Check for successful response
        if response.status == 200:
            html = await response.text()
            return html
        return None

async def scrape(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.title
    # Check if the title tag exists and return its text; otherwise, return a default message
    return title_tag.string if title_tag else "No title found"

async def main():
    # Scrape the URLs
    html_responses = await scrape(urls)
    results = []

    for html in html_responses:
        if html:
            result = parse_html(html)
            results.append(result)

    # Save results to a text file
    with open('/content/scraped_data.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')

# Run the main function using asyncio.run() in the already running event loop
await main()

# Download the scraped data file
from google.colab import files
files.download('/content/scraped_data.txt')


________________________________________________________________________________________

#!pip install aiohttp nest_asyncio

import asyncio
import aiohttp
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# List of API endpoints to call
api_urls = [
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
    # Add more API endpoints as needed
]

async def fetch(session, url):
    async with session.get(url) as response:
        # Check for successful response
        if response.status == 200:
            data = await response.json()  # Parse response as JSON
            return data
        return None

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in api_urls]
        results = await asyncio.gather(*tasks)

        # Process and print the results
        for result in results:
            if result is not None:
                print(result)
            else:
                print("Failed to fetch data.")

# Run the main function
await main()


________________________________________________________________________________

#!pip install aiofiles

import asyncio
import aiofiles

# Function to write data to a file asynchronously
async def write_large_file(filename, data):
    async with aiofiles.open(filename, mode='w') as f:
        await f.write(data)

# Function to read data from a file asynchronously
async def read_large_file(filename):
    async with aiofiles.open(filename, mode='r') as f:
        content = await f.read()
        return content

async def main():
    filename = '/content/scraped_data.txt'

    # Simulate writing large data
    large_data = "This is a line in the file.\n" * 10000  # Example large data
    await write_large_file(filename, large_data)
    print(f"Written to {filename}.")

    # Read the data back from the file
    content = await read_large_file(filename)
    print(f"Read from {filename}: {content[:100]}...")  # Print the first 100 characters

# Run the main function
await main()
________________________________________________________________


!pip install sqlalchemy

import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# Create a SQLite database connection
conn = sqlite3.connect('example.db')

# Create a cursor object
cursor = conn.cursor()

# Create a table
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    department TEXT NOT NULL
)
''')

# Function to insert data into the database
def insert_data(data):
    cursor.executemany('''
    INSERT INTO employees (name, age, department) VALUES (?, ?, ?)
    ''', data)
    conn.commit()

# Generate sample data
sample_data = [
    ("Alice", 30, "HR"),
    ("Bob", 24, "Engineering"),
    ("Charlie", 28, "Marketing"),
    ("David", 35, "Engineering"),
    ("Eve", 29, "HR"),
    ("Frank", 32, "Marketing")
]

# Insert sample data into the database
insert_data(sample_data)

# Function to query data from the database
def query_data():
    return pd.read_sql_query('SELECT * FROM employees', conn)

# Query and display the data
dataframe = query_data()
print("Data before optimization:")
print(dataframe)

# Optimize by creating an index on the department column
cursor.execute('CREATE INDEX IF NOT EXISTS idx_department ON employees(department)')

# Query data again to show optimization effect
optimized_dataframe = pd.read_sql_query('SELECT * FROM employees WHERE department = "Engineering"', conn)
print("\nData after optimization (filtered by department):")
print(optimized_dataframe)

# Clean up and close the connection
conn.close()

__________________________________________________________________


