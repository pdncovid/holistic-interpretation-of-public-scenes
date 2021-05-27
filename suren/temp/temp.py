from suren.util import  Json
import json
# import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

n = int(input())

if n == 0:
    while 1:
        pass

arr = []
for x in range(n):
    a = [int(i) for i in input().strip().split()]
    

    if x > 0:
        a[0] += arr[0]
        a[-1] += arr[-1]
        for j in range(1, x):
            a[j] += max(arr[j - 1:j + 1])

    arr = a[:]

    print(arr)

print(max(arr))