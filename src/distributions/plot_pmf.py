import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt

figure_dir = Path('../../reports/figures')
subdir = figure_dir / 'distributions' / dt.datetime.now().strftime('%Y-%m-%d')
if not subdir.exists():
    subdir.mkdir(parents=True)

# geometric distribution
p = 0.3

# pmf
k = np.arange(1,20)
y = sp.stats.geom.pmf(k, p)
f, ax = plt.subplots()
plt.plot(k, y, 'o-', label="p={}".format(p))
plt.title("Geometric distribution pmf, p={}".format(p))
f.savefig(subdir / 'geometric_pmf.png')
# plt.show()

# simulate
N = 5000
runs = []
for _ in range(N):
    i = 0
    while True:
        i += 1
        if np.random.rand() < p:
            runs.append(i)
            break

f, ax = plt.subplots()
plt.hist(runs, bins=20, density=True)
plt.title("Geometric distribution histogram, p={}".format(p))
f.savefig(subdir / 'geometric_histogram.png')

# cdf
f, ax = plt.subplots()
y = sp.stats.geom.cdf(k, p)
plt.plot(k, y, 'o-', label="p={}".format(p))
plt.title("Geometric distribution cdf, p={}".format(p))
f.savefig(subdir / 'geometric_cdf.png')

# plt.show()



