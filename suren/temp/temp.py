from suren.util import  Json
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

# with open('./nn-outputs/yoloOut.txt') as file:
#     data = file.read()
#     data = json.dump


plt.ion()
x = np.arange(0, 4*np.pi, 0.1)
y = [np.sin(i) for i in x]
plt.plot(x, y, 'g-', linewidth=1.5, markersize=4)
plt.draw()
time.sleep(1)
plt.plot(x, [i**2 for i in y], 'g-', linewidth=1.5, markersize=4)
plt.pause(1)
plt.plot(x, [i**2*i+0.25 for i in y], 'r-', linewidth=1.5, markersize=4) 
plt.pause(1)

plt.show(block=True)

