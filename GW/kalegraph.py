#!/usr/bin/env python3.12
import kalepy as kale
import matplotlib.pyplot as plt
points, density = kale.density(data3, points=None)
plt.plot(points,density, 'k-', lw=2.0, alpha=0.8, label='KDE')
