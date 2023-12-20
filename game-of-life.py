
# ** CELLULAR AUTOMATA - GAME OF LIFE
# ** Author: Franck Delaplace
# ** MASTER TUTORIAL
# ** Paris Saclay University

from cellularautomata import CountType, GuiCA


# A cell of CA is composed by a tuple (type, states ...) where:
#  - type indicates the category/type of the cell which must be a string
#  - states indicate the internal states of the cell which depends on the simulation.
# A dictionary (cellcolors) stores the initial cell configurations as keys with the related color as value for each type.

# The evolution function has two parameters: the cell and the list of neighboring cells.

# GAME OF LIFE RULE
# A dead cell with exactly three living neighbors become alive (is born)
# A living cell with two or three living neighbors remains so.
# Otherwise, the cell dies or stay died.
# for GoL the cell type (name) suffices therefore the state is useless and set to None.

def GoL(cell, neighbors: list):             # Game of life evolution function for a cell.
    type, _ = cell
    alive = CountType(neighbors, 'Alive')   # count the number of alive cells
    if alive == 2 and type == 'Alive':      # 2 neighbors and being alive => still alive.
        return ('Alive', None)
    elif alive == 3:                        # 3 neighbors => being alive.
        return ('Alive', None)
    else:					                # otherwise the cell dies.
        return ('Dead', None)


# Main program ============================================================
cellcolors = {('Dead', None): 'white', ('Alive', None): 'black'}  # color assigned to cells
GuiCA(GoL, cellcolors, gridsize=100, duration=200)

# MANUAL RUN
# from cellularautomata import GenerateCA, SimulateCA, ShowSimulation
# weights =  {'Alive': 0.35, 'Dead': 0.65}
# CA0 = GenerateCA(50, cellcolors, weights)
# simulation = SimulateCA(CA0, GoL, duration=100)
# animation = ShowSimulation(simulation, cellcolors)
