# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:29:02 2023

@author: ayaha
"""
from numpy import *
from matplotlib.pyplot import *

fig = figure()

ax =fig.add_subplot(111, projection = '3d')

x1 = arange(-1,1,0.1)
y1 = arange(-1,1,0.1)

x1,y1=meshgrid(x1,y1)

z1=x1**2 + y1**2 - (0.45**2)

ax.plot_surface(x1, y1, z1, color ="black")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
show()

"""
xp1=linspace(-10,10,100)
yp1=linspace(-10,10,100)
xp1,yp1 = meshgrid(xp1, yp1)
eq1 = 0.12 * xp1 + 0.01 * yp1 + 1.09
"""

"""
    Circle z^2 + y^2 = 0.45^2
    z = 0.45 - y
    Line1 
    Line2 
    Line3


"""