
# ** CELLULAR AUTOMATA - FOREST FIRE
# ** Author: Franck Delaplace
# ** MASTER TUTORIAL
# ** Paris Saclay University

from cellularautomata import CountType, GuiCA
from random import random

# The simulation studies the spread of a fire in a forest.
# If a tree catches fire, there is a certain probability that neighboring trees will also burn. (e.g. 0.2).
# The probability that a tree will be on fire is proportional to the number of neighboring burning trees.
# After a certain time, a burning tree turns to ash.

TFIRE: int = 7          # Fire period before turning to ash.
ONFIRE: float = 0.2      # Probability to be on fire.


def FoF(cell, neighbors: list):
    category, time = cell
    match category:
        case 'Tree':
            if random() < CountType(neighbors, 'Fire') * ONFIRE:
                return ('Fire', TFIRE)
            else:
                return cell
        case 'Fire':
            if time == 0:
                return ('Ash', None)
            else:
                return ('Fire', time - 1)
        case _:
            return cell


# Main program ============================================================
cellcolors = {('Tree', None): 'forestgreen', ('Soil', None): 'white', ('Fire', TFIRE): 'crimson', ('Ash', None): 'darkgray'}  # color assigned to states

GuiCA(FoF, cellcolors)
