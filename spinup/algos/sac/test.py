import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = np.arange(10)
    plt.figure()
    plt.plot(x,x+1)
    plt.show()
