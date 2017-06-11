import sys
import matplotlib.pyplot as plt

def plot(filename):
    values = []
    with open(filename) as f:
        for line in f:
            if 'acc:' in line:
                values.append(float(line.strip().split()[-1]))
    return values

if __name__ == '__main__':
    filename = sys.argv[1]
    values = plot(filename)
    if len(sys.argv) > 2:
        last = int(sys.argv[2])
    else:
        last = len(values)
    plt.plot(values[-last:])
    plt.show()