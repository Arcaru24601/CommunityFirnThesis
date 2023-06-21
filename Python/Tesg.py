# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:55:48 2023

@author: Jesper Holm
"""

import matplotlib.pyplot as plt
import numpy as np




t = np.arange(0, 10, 0.01)

y1 = 2*np.pi*t
y2 = 4*np.pi*np.sqrt(t)
y1 += np.random.normal(20,10,1000)
y2 += np.random.normal(5,30,1000)

fig, ax = plt.subplots(nrows=2,sharex=True)


ax[1].plot(t, y1,'r')
ax[0].plot(t, y2,'b')



twin1 = ax[1].twinx()
twin2 = ax[0].twinx()

twin1.plot(t,y2,'g')
twin2.plot(t,y1,'y')













#p1, = ax.plot(x,y, "C0", label="Density")
#p2, = twin1.plot(x, y1, "C1", label="Temperature")
#p3, = twin2.plot(x, y2, "C2", label="Velocity")



ax[1].yaxis.label.set_color('r')
ax[0].yaxis.label.set_color('b')
twin1.yaxis.label.set_color('g')
twin2.yaxis.label.set_color('y')

ax[1].tick_params(axis='y', colors='r')
ax[0].tick_params(axis='y', colors='b')
twin1.tick_params(axis='y', colors='g')
twin2.tick_params(axis='y', colors='y')

#ax[1].spines.left.set_bounds(y1.min(), y1.max())
#ax[0].spines.left.set_bounds(y2.min(), y2.max())

#ax.spines.bottom.set_bounds(x.min(), x.max())
twin2.spines.right.set_bounds(y1.min(), y1.max())
twin1.spines.right.set_bounds(y2.min(), y2.max())
ax[0].tick_params(axis='x', which='both', bottom=False)

#ax.spines.left.set_bounds(y.min(), y.max())


#twin1.set_yticks([0,1,2,3])

#twin2.set_yticks([15, 27,38, 50])


# Hide the right and top spines
#twin2.spines.right.set_visible(False)
#twin2.spines.top.set_visible(False)

ax[1].spines.right.set_visible(False)
ax[0].spines.bottom.set_visible(False)
ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
ax[1].spines.top.set_visible(False)



twin1.spines.top.set_visible(False)
twin2.spines.top.set_visible(False)
twin1.spines.left.set_visible(False)
twin2.spines.left.set_visible(False)
#ax.spines.top.set_visible(False)
twin2.spines.bottom.set_visible(False)

#ax.legend(handles=[p1, p2, p3])

plt.show()