
# ** CELLULAR AUTOMATA LIBRARY
# ** Author: Franck - 2023
# ** MASTER TUTORIAL
# ** Paris Saclay University

from matplotlib.animation import FuncAnimation, PillowWriter
from random import choices
import numpy as np
import matplotlib.colors as color
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.patches import Rectangle


def CountType(cells: list, category: str) -> int:
    """Return the number of cells whose type matches with the category in a list of cells.

    Args:
        cells (list): list of cells
        category (str): expected type

    Returns:
        int: number of types matching with the category
    """
    return [category for category, *_ in cells].count(category)


def GenerateCA(n: int, cellcolors: dict, weights: dict | None = None) -> np.ndarray:
    """Generate a n*n 2D cellular automaton randomly.
    Args:
    n (int): height and width of the grid
    cellcolors (dict): cell types with their associated cellcolors
    weights (dict | None, optional): probabilistic weights of the values. Defaults to None.

    Returns: (n,n) 2D cellular automaton.
    """
    assert n > 0
    cells = list(cellcolors.keys())  # Get cell definitions

    if weights is None:
        weights_ = None
    else:
        weights_ = [weights[category] for category, *_ in cells]    # Collect the weights in list from the weight dictionary.

    randca = choices(cells, weights=weights_, k=n * n)                          # Generate the cells randomly
    return np.array([[randca[i + n * j] for i in range(n)] for j in range(n)])  # Reshape to get a 2D array


def DrawCA(cellautomaton: np.ndarray, colors: list, ax):
    """Draw a 2D cellular automaton

    Args:
        CA (np.array): Cellular Automata
        ax : axes
        colors (list): list of colors

    Returns: a graphical view
    """
    return sns.heatmap(
        cellautomaton,
        cmap=color.ListedColormap(colors),
        linewidths=0.5,
        cbar=False,
        linecolor="lightgrey",
        clip_on=False,
        vmin=-0.5,
        vmax=len(colors) - 0.5,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )


def SimulateCA(cellautomaton0: np.ndarray, f, numsteps: int = 100) -> list:
    """Compute a simulation of a cellular automaton.
    Args:
        cellautomaton0 (np.ndarray): initial cellular automata
        f (fun): local update function
        numsteps (int, optional): total number of steps. Default 100.

    Returns:
        list: Execution trace corresponding to a list of cellular automata.
    """
    assert numsteps >= 0

    def ca_step(cellautomaton: np.ndarray, f) -> np.ndarray:  # Compute 1 CA step.
        global _local_value
        MOORE = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]  # Displacement of the Moore neighborhood
        n = len(cellautomaton)
        mooreshift = np.array([np.roll(cellautomaton, dis, axis=(0, 1)) for dis in MOORE])  # Copies of CA cyclically shifted according to Moore's neighborhood
        neighborsgrid = list(np.transpose(mooreshift, axes=(1, 2, 0, 3)))                   # Transposition to obtain a 2D array of neighbor lists
        canew = np.array(
                [[f(cellautomaton[i][j], neighborsgrid[i][j]) for j in range(n)] for i in range(n)]
                )  # apply the local evolution function

        return canew

    simulation = [cellautomaton0]
    try:
        for i in range(numsteps):
            simulation.append(ca_step(simulation[i], f))
    except ValueError:
        print("** CA ERROR: Invalid cell format encountered. A condition on cell is probably missing in the local function.")
        exit()  # End program

    return simulation


# Switch for managing the autorun in ShowSimulation.
class Switch:
    "Boolean value toggling for switch control."
    state: bool

    def __init__(self, val: bool = True):
        self.state = val

    def switch(self):          # Toggle the state.
        self.state = not self.state
        return self.state

    def get(self):             # Get the state.
        return self.state

    def set(self, val: bool):  # Set the state.
        self.state = val


_animation = None       # Variable storing the visualization, must be global.
_autorun_button = None  # Button autorun ON/OFF, must be global to properly work.
_save_button = None     # Button to save Simulation, must be global to properly work.
_curve_button = None    # CheckBox Button for curves, must be global to properly work.

def ShowSimulation(simulation: list, cellcolors: dict[tuple, str], figsize: int = 5, delay: int = 100):
    """Display the simulation trace of a cellular automaton.

    Args:
        simulation (list):  simulation trace
        cellcolors (dict): colors assigned to cells
        figsize (int, optional): size of the graphical window. Defaults to 5.
        delay (int, optional): delay in ms between two steps. Defaults to 100.

    Returns:
        _type_: animation
    """
    assert delay > 0
    assert figsize > 0

    global _autorun_button
    global _save_button
    global _curve_button
    global _animation

    # Preamble
    n = len(simulation)
    autorun = Switch()

    # Figure definition
    plt.rcParams["font.family"] = "fantasy"  # 'monospace'  'sans'
    plt.rcParams["font.size"] = 11
    plt.rcParams["text.color"] = "black"

    figtitle = "CELLULAR AUTOMATON - FD MASTER COURSE"

    if plt.fignum_exists(figtitle):  # If a new simulation is launched without closing the window then close it.
        plt.close(figtitle)

    fig = plt.figure(figtitle, figsize=(2 * figsize, figsize), dpi = 100)
    wmgeometry = "+400+150"
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry(wmgeometry)
    # Order the colors to suit the DrawCA function w.r.t. to the types.
    cells = list(cellcolors.keys())                 # Extract cells from cellcolors
    cells.sort()                                    # The order of the cells follow the order of the types since the type is at first.
    colors = [cellcolors[cell] for cell in cells]   # Extract the color following the order of the types
    types = [category for category, *_ in cells]    # Extract types ordered.

    # Axe of CA + initialization of the CA display.
    X0 = 0.02  # Left bottom position of the CA
    Y0 = 0.1
    axca = fig.add_axes([X0, Y0, 0.45, 0.9])

    # CA initialization where the cells are encoded by their index of type in types to properly suit with colors.
    ca_coded = np.array([[types.index(category) for category, *_ in row] for row in simulation[0]])
    caview = DrawCA(ca_coded, colors, axca).collections[0]

    # Axe of curves
    CHEIGHT = 0.87  # Height of the curve axe.
    axcurve = fig.add_axes([X0 + 0.52, Y0, 0.44, CHEIGHT])
    axcurve.set_xlim(0, n)
    axcurve.set_ylim(0, len(simulation[0]) ** 2)
    axcurve.grid(linestyle="--")

    # Initialize the count curves.
    typescount = {
        category: [sum([CountType(row, category) for row in ca]) for ca in simulation]
        for category in types}

    visible_curves = [color != "white" for color in colors]  # All curves are visible but the white ones.
    curves = {                                               # The curves are collected to a dictionary {type: counting curve}.
        category: axcurve.plot(
            [0],
            typescount[category][0],
            color=colors[i] if colors[i] != "white" else "lightgray",  # The white color is transformed into a light gray
            linewidth=2.5,
            marker=" ",
            visible=visible_curves[i],
        )[0]
        for i, category in enumerate(types)}

    # Check box button characterization for curves
    chxboxheight = len(types) * 0.05                    # Depends on the number of categories.
    chxboxwidth = 0.05 + max(map(len, types)) * 0.006   # Depends on the maximal string length of the categories.
    axcurvebox = fig.add_axes(
        [X0 + 0.52, Y0 + CHEIGHT - chxboxheight, chxboxwidth, chxboxheight])    # The check boxes are located in the upper left of the curve graphics.

    _curve_button = CheckButtons(axcurvebox, types, visible_curves)

    def chxboxupdate(category: str) -> bool:  # update the check boxes
        return curves[category].set_visible(not curves[category].get_visible())  # Toggle the visibility of curve.
    _curve_button.on_clicked(chxboxupdate)

    # Slider characterization
    axslider = fig.add_axes([X0 + 0.04, Y0 - 0.07, 0.412, 0.07])    # The slider is located below the cellular automaton display.
    slider = Slider(axslider, "", 0, n - 1, valstep=1, valinit=0, facecolor="gray", valfmt="%3d")

    xrange = np.arange(0, n, 1, dtype=int)

    def updateslider(step):  # Update of slider.
        CAcode = np.array([[types.index(category) for category, *_ in row] for row in simulation[step]])
        caview.set_array(CAcode)    # Update CA
        for category in types:              # Update type count curves
            curves[category].set_data(xrange[:step], typescount[category][:step])
        return curves
    slider.on_changed(updateslider)  # Event on slider.

    # ON/OFF autorun Button.
    ax_autorun_button = fig.add_axes([X0+0.02, Y0 - 0.05, 0.015, 0.03])  # ON/OFF button is on the left side of slider.
    _autorun_button = Button(ax_autorun_button, " ")

    # Button labeling to indicate autorun status.
    OFF_ICON = "$\u25a0$"  # square
    ON_ICON = "$\u25B6$"   # right triangle
    def buttonlabeling(state: bool):  # Set the label ON/OFF to the button w.r.t. to a Boolean state.
        _autorun_button.label.set_text({False: ON_ICON, True: OFF_ICON}[state])

    buttonlabeling(autorun.get())  # Initialize button label from the initial autorun state.

    def click_autorun_button(_):
        global _autorun_button
        autorun.switch()                                # Switch the autorun.
        buttonlabeling(autorun.get())                   # Change the button label.
    _autorun_button.on_clicked(click_autorun_button)    # Event on autorun button.

    # Button save Animation
    ax_save_button = fig.add_axes([X0, Y0 - 0.05, 0.015, 0.03])  # ON/OFF button is on the left side of slider.
    SAVED_ICON = "$\u25BD$"  # triangle pointing down, empty shape
    SAVE_ICON = "$\u25BC$"   # triangle pointing down, filled shape
    _save_button = Button(ax_save_button, SAVE_ICON)
    saved = Switch(False)

    def click_save_button(_):
        global _save_button
        writer = PillowWriter(fps=1500//delay)
        _animation.save("CA-SIMULATION.gif", writer=writer)
        msgput("Save completed!")
        saved.set(True)
        _save_button.label.set_text(SAVED_ICON)
    _save_button.on_clicked(click_save_button)  # Event on save button.

    # Tooltips handler
    axmsg = fig.add_axes([X0, Y0 - 0.09, 0.45, 0.03], facecolor="gainsboro")   # The message zone is below the slider

    def msgclear():  # Clear the message box.
        axmsg.cla()
        axmsg.set_xticks([])
        axmsg.set_yticks([])

    def msgput(msg: str):  # Print a message in the message box.
        msgclear()
        axmsg.text(0.01, 0.2, msg, fontsize=8, fontfamily='serif', fontstyle='italic')

    # handling events
    def hover(event):  # Event over axes handler
        if ax_save_button.contains(event)[0]:
            if saved.get():
                msgput("Click to save the simulation in GIF - Simulation already saved.")
            else:
                msgput("Click to save the simulation in GIF.")
        elif ax_autorun_button.contains(event)[0]:
            msgput("Click to turn ON/OFF the simulation: "+OFF_ICON+" = OFF, "+ON_ICON+" = ON.")
        elif axca.contains(event)[0]:
            msgput("Cellular Automaton.")
        elif axcurve.contains(event)[0]:
            msgput("Type count curves.")
        else:
            msgclear()

    def onclick(event):  # Click on axes handler.
        if ax_save_button.contains(event)[0]:
            msgput("Save in progress")
        elif ax_autorun_button.contains(event)[0]:
            msgput("Simulation switched "+("OFF, scroll the slider." if autorun.get() else "ON."))
        else:
            pass

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", onclick)

    msgclear()

    # Display simulation
    def updateanimation(_):         # Update from animation.
        if autorun.get():           # The update is conditional on the state of autorun.
            step = (slider.val + 1) % slider.valmax
            slider.set_val(step)    # Updating slider value also triggers the updateslider function

    _animation = FuncAnimation(fig, updateanimation, interval=delay, save_count=n)  # Run animation.
    plt.show()
    return _animation


# Cellular Automaton graphical user  interface.

# Class to manage weights for random definition of the CA grid
class Weights:
    "Weights = dictionary associating cells to their weights which are floats between 0 and 1"
    weights: dict = {}

    def __init__(self, types: list, value: float = 0.0):
        self.weights = {state: value for state in types}

    def set(self, state: int, val):  # Set the weight of a state.
        self.weights[state] = val

    def get(self, state):  # Get the weight of a state.
        return self.weights[state]

    def check(self):  # Check whether the weights are consistent. i.e. 0 <= w <= 1.
        isweight = True
        for w in self.weights.values():
            isweight = isweight and 0.0 <= w <= 1.0
        return isweight


# Global variables used for passing parameters to sliders and buttons
_gridsize = 1       # CA grid size
_duration = 1       # Duration of the simulation


def GuiCA(
    local_fun,
    cellcolors: dict,
    figsize: int = 5,
    gridsize: int = 100,
    duration: int = 200,
    delay: int = 100
):
    """Graphical interface for cellular Automata.
        limited to 10 types at most.

    Args:
        local_fun (function): local update function of the CA.
        cellcolors (dict): {cell:color} colors associated to cells. Recall that a cell is a tuple (type, states ..)
        figsize (int, optional): size of the figure of the simulation view. Defaults to 5.
        gridsize (int, optional): maximal size of the CA grid. Defaults to 100.
        duration (int, optional): maximal duration of the simulation. Defaults to 200.
        delay (int, optional): delay in ms between two simulation steps. Defaults to 100.
    """
    assert all([isinstance(cell, tuple) for cell in cellcolors])   # Check that keys are tuples!
    assert all([isinstance(category, str) for category, *_ in cellcolors])  # check that the types are strings!
    assert len(cellcolors) <= 10  # limited to 10 parameters - see program to understand this limitation.
    assert figsize > 0
    assert gridsize > 0
    assert duration > 0

    global _gridsize
    global _duration
    # Windows parameters
    GUIWIDTH = 1.5      # Width of the GUI
    GUISTEP = 0.15      # Extra width step associated to characters of type labels.
    GUIHEIGHT = 4       # Height of the GUI

    # Button & Slider parameters
    SLIDLEFT = 0.35     # Left position of sliders
    SLIDSIZE = 0.4      # Size of sliders
    SLIDHEIGHT = 0.07   # Height of sliders
    SLIDSTART = 0.7     # Vertical start position for weight sliders
    SLIDDIST = 0.05     # Distance between two weight sliders
    SLIDCOLOR = "gray"  # Slider color bar

    # Initialization of the main variables
    _gridsize = gridsize // 2
    _duration = duration // 2
    types = [type for type, *_ in cellcolors.keys()]  # get all types of cells
    types.sort()  # sort types
    n = len(types)
    weights = Weights(types, 0.5)  # Create weights from types.

    # Initialization of the figure
    plt.rcParams["toolbar"] = "None"    # No tool bars on GUI figure.
    plt.rcParams["font.family"] = "sans"
    plt.rcParams["font.size"] = 8

    fig = plt.figure(figsize=(GUIWIDTH + max(map(len, types)) * GUISTEP, GUIHEIGHT), num="GUI")
    ax = fig.add_axes([0, 0, 1, 1])
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("+50+100")

    # Grid size slider ======
    axsize_slider = plt.axes([SLIDLEFT, 0.92, SLIDSIZE, SLIDHEIGHT])
    size_slider = Slider(
        axsize_slider,
        "Size  ",
        1,
        gridsize,
        valstep=1,
        valinit=_gridsize,
        facecolor=SLIDCOLOR,
        valfmt="%3d",
    )

    def update_slider_size(val: int):
        global _gridsize
        _gridsize = val
    size_slider.on_changed(update_slider_size)  # Event on size slider

    # Duration/Time sliders ======
    axduration_slider = plt.axes([SLIDLEFT, 0.86, SLIDSIZE, SLIDHEIGHT])
    duration_slider = Slider(
        axduration_slider,
        "Time  ",
        1,
        duration,
        valstep=1,
        valinit=_duration,
        facecolor=SLIDCOLOR,
        valfmt="%3d",
    )

    def update_slider_duration(val: int):
        global _duration
        _duration = val
    duration_slider.on_changed(update_slider_duration)  # Event on duration slider

    # Weights  sliders ======

    # header and rectangle
    FRMLEFT = 0.07  # Frame left position
    FRMSIZE = 0.86  # Frame size
    font = {"weight": "bold", "size": 10}
    ax.text(0.5, 0.79, "WEIGHTS", fontdict=font, ha="center")
    ax.add_patch(
        Rectangle(
            (FRMLEFT, 0.15),
            FRMSIZE,
            0.7,
            facecolor="whitesmoke",
            edgecolor="darkgray",
            linewidth=2,
        )
    )

    # Weight sliders definition
    weight_sliders = []
    for i in range(n):
        axslider = plt.axes(
            [SLIDLEFT, SLIDSTART - i * SLIDDIST, SLIDSIZE, SLIDHEIGHT],
            facecolor=SLIDCOLOR,
        )
        slider = Slider(
            axslider,
            str(types[i]) + "  ",
            0.0,
            1.0,
            valstep=0.005,
            valinit=0.5,
            facecolor=SLIDCOLOR,
            valfmt="%1.3f",
        )
        weight_sliders.append(slider)

    # All possible updates for weight slide from 0 to 9
    # ! I don't find a better solution avoiding to limit the number of parameters to assign the update functions to each weight slider.
    # ! [lambda val:weights.set(types[i],val) for i in range(n)]  DOES NOT WORK (i = max for all buttons !?)
    weight_update_fun = [
        lambda val: weights.set(types[0], val),
        lambda val: weights.set(types[1], val),
        lambda val: weights.set(types[2], val),
        lambda val: weights.set(types[3], val),
        lambda val: weights.set(types[4], val),
        lambda val: weights.set(types[5], val),
        lambda val: weights.set(types[6], val),
        lambda val: weights.set(types[7], val),
        lambda val: weights.set(types[8], val),
        lambda val: weights.set(types[9], val),
    ]

    for i in range(n):  # Link events to weight sliders
        weight_sliders[i].on_changed(weight_update_fun[i])

    # Run Button ======
    axrun_button = plt.axes([FRMLEFT, 0.05, FRMSIZE, SLIDHEIGHT])
    run_button = Button(axrun_button, "RUN", color="silver", hovercolor="lightsalmon")

    def runclick(_):  # Run Button clicked
        global _gridsize
        global _duration
        global _animation

        # check if at least a weight is different to 0
        if sum(weights.weights.values()) == 0:
            print("** CA WARNING: at least one weight must be different to 0.")
        else:
            CA = GenerateCA(_gridsize, cellcolors, weights.weights)
            simulation = SimulateCA(CA, local_fun, numsteps=_duration)
            _animation = ShowSimulation(simulation, cellcolors, figsize=figsize, delay=delay)

    run_button.on_clicked(runclick)  # Event on button

    plt.show()
