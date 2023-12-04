    
from matplotlib.animation import FuncAnimation
from random import choices
import numpy as np
import matplotlib.colors as color
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

X0 = 0.5  # left bottom position of the CA
Y0 = 0.2
figsize = 5
fig = plt.figure("CELLULAR AUTOMATON - FD MASTER COURSE", figsize=(figsize, figsize))
ax = plt.gca()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
  
ax_save_button = plt.axes([X0, Y0 - 0.05, 0.2, 0.05])  # ON/OFF button is on the left side of slider.

SAVED_ICON = "$\u25BD$"  # triangle pointing down, empty shape
SAVE_ICON = "$\u25BC$"   # triangle pointing down, filled shape
_save_button = Button(ax_save_button, SAVE_ICON)

def click_save_button(_):
    global _save_button
    _save_button.label.set_text(SAVED_ICON)
    print("** Save simulation to 'CA-SIMULATION.gif' file.")
    print("** Backup Completed !")
    _save_button.on_clicked(click_save_button)  # Event on save button.


    # Tooltip Annotation 
def hover_annotate(event):
    if ax_save_button.contains(event)[0]:
        print("*" , end=" ")

fig.canvas.mpl_connect('motion_notify_event', hover_annotate) 

plt.show()