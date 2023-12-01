
# ** CELLULAR AUTOMATA - GAME OF LIFE
# ** Author: Franck Delaplace - 2023
# ** MASTER TUTORIAL
# ** Paris Saclay University

from cellularautomata import CountType, GuiCA


# A cell of CA is composed by a tuple (type, states ...) where:
#  - type indicates the category/type of the cell - usually a string
#  - states indicate the internal states of the cell which depends on the simulation.

# GAME OF LIFE RULE
# A dead cell with exactly three living neighbors becomes alive (is born)
# A living cell with two or three living neighbors remains so.
# Otherwise, the cell dies or stay died.
# for GoL the cell type (name) suffices therefore the state is useless and set to None

def GoL(cell, neighbors: list):
    TYPE = 0
    type, _ = cell
    alive = CountType(neighbors, 'Alive')   # count the number of alive cells
    if alive == 2 and type == 'Alive':      # 2 neighbors and being alive => still alive.
        return ('Alive', None)
    elif alive == 3:                        # 3 neighbors => being alive.
        return ('Alive', None)
    else:					                # otherwise the cell dies.
        return ('Dead', None)


# Main program ============================================================
cellcolors = {('Dead', None): 'white', ('Alive', None): 'black'}  # color assigned to states
GuiCA(GoL, cellcolors, gridsize=100, duration=200)

# MANUAL RUN
# from cellularautomata import GenerateCA, SimulateCA, ShowSimulation
# n = 50
# weights =  {'Alive':0.35, 'Dead': 0.65}
# CA0= GenerateCA(50, cellcolors, weights )
# simulation = SimulateCA(CA0, GoL)
# animation = ShowSimulation(simulation, cellcolors)
