import numpy as np
import matplotlib.pyplot as plt
import sys

""" cat temp.txt | py plot.py """

def main():

    costs = []
    for line in sys.stdin:
        costs.append(float(line.rstrip()))


    plt.semilogy(range(len(costs)), costs, 'k-o')

    plt.show()
    return

if __name__ == '__main__':
    main()
