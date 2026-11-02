import os
import csv
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, 'trajectory_save.csv'))))
x = [float(r['x']) for r in rows]
y = [float(r['y']) for r in rows]

plt.plot(x, y, 'r.', markersize=8)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(HERE, 'trajectory.png'), dpi=200)
plt.show()