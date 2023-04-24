# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:30:51 2023

@author: Jesper Holm
"""

from multiprocessing import Process


def add(a, b):
    total = 0
    for a1, b1 in zip(a, b):
        total = a1 + b1
    return total


def sub(s, t):
    total = 0
    for a1, b1 in zip(s, t):
        total = a1 - b1
    return total


def mult(y, x):
    total = 0
    for a1, b1 in zip(y, x):
        total = a1 * b1
    return total


if __name__ == "__main__":
    # construct a different process for each function
    max_size = 1000000000
    processes = [Process(target=add, args=(range(1, max_size), range(1, max_size))),
                 Process(target=sub, args=(range(1, max_size), range(1, max_size))),
                 Process(target=mult, args=(range(1, max_size), range(1, max_size)))]

    # kick them off 
    for process in processes:
        process.start()

    # now wait for them to finish
    for process in processes:
        process.join()