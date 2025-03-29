# %%

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3*np.pi, 750)

plt.plot(x,np.sin(x), linewidth = 8)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)


plt.show()