import sys
import matplotlib.pyplot as plt

def plot(filename):
    values = []
    with open(filename) as f:
        values.extend(float(line.strip().split()[-1]) for line in f if 'acc:' in line)
    return values

if __name__ == '__main__':
    filename = sys.argv[1]
    values = plot(filename)
    last = int(sys.argv[2]) if len(sys.argv) > 2 else len(values)
    plt.plot(values[-last:])
    plt.show()