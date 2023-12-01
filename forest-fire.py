
# ** CELLULAR AUTOMATA - FOREST FIRE
# ** Author: Franck Delaplace - 2023
# ** MASTER TUTORIAL
# ** Paris Saclay University

from cellularautomata import CountType, GuiCA
from random import random

# The simulation studies the spread of a fire in a forest.
# When a tree is on fire, neighboring trees can burn with a given probability (e.g. 0.2).
# The likelihood of being burned for a tree is proportional to the number of neighboring trees in fire.
# After a certain time, a tree turns to ash.

TIME = 1
TFIRE = 7
FIRING = 0.2

def FoF(cell, neighbors: list):
    category, _ = cell
    match category:
        case 'Tree':
            if  random() < CountType(neighbors, 'Fire') * FIRING:
                return ('Fire', TFIRE)
            else:
                return(cell)
        case 'Fire':
            if cell[TIME] == 0:
                return ('Ash', None)
            else:
                return ('Fire', cell[TIME] - 1)
        case _ :
            return cell


# Main program ============================================================
cellcolors = {('Tree', None): 'forestgreen', ('Fire', TFIRE): 'crimson', ('Ash', None): 'darkgray', ('Grass', None): 'white'}  # color assigned to states

GuiCA(FoF, cellcolors)
