import time
import psutil
import os


def log_cost(start_time):
    end_time = time.time()
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 ** 2

    print("Inference Time:", end_time - start_time, "seconds")
    print("Memory Usage:", memory, "MB")
