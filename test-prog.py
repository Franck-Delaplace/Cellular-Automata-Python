
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Créer une fonction d'animation
def animate(frame):
    # Votre code d'animation ici
    # Par exemple, pour un tracé en fonction du temps
    plt.plot(frame, frame ** 2, 'ro-')
    plt.title('Animation Example')

# Configurer la figure
fig, ax = plt.subplots()
frames = 10  # Nombre de frames dans l'animation

# Créer l'animation
anim = FuncAnimation(fig, animate, frames=frames, interval=200)

writer = PillowWriter(fps=30)
anim.save("CAex.gif", writer=writer)

# Afficher l'animation (facultatif)
plt.show()