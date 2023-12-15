
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
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons, RectangleSelector
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

    Returns: a heatmap representing the CA.
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


def SimulateCA(cellautomaton0: np.ndarray, f, duration: int = 100) -> list:
    """Compute a simulation of a cellular automaton.
    Args:
        cellautomaton0 (np.ndarray): initial cellular automata
        f (fun): local update function
        duration (int, optional): total number of steps. Default 100.

    Returns:
        list: Simulation trace corresponding to a list of cellular automata.
    """
    assert duration >= 0

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
        for i in range(duration):
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
        return val


_animation = None       # Variable storing the visualization, must be global.
_autorun_button = None  # Button autorun ON/OFF, must be global to properly work.
_save_button = None     # Button to save Simulation, must be global to properly work.
_curve_button = None    # CheckBox Button for curves, must be global to properly work.


def ShowSimulation(simulation: list, cellcolors: dict[tuple, str], figheight: int = 5, delay: int = 100):
    """Display the simulation trace of a cellular automaton.

    Args:
        simulation (list):  simulation trace
        cellcolors (dict): colors assigned to cells
        figheight (int, optional): height of the figure with figure size = (2*figheight,figheight). Defaults to 5.
        delay (int, optional): delay in ms between two steps. Defaults to 100.

    Returns:
        _type_: animation
    """
    assert delay > 0
    assert figheight > 0

    global _autorun_button
    global _save_button
    global _curve_button
    global _animation

    # Preamble
    n = len(simulation)
    autorun = Switch()

    # Figure definition
    figtitle = "CELLULAR AUTOMATON - FD MASTER COURSE"  # Feel free to change the title.

    plt.rcParams["font.family"] = "fantasy"  # 'monospace'  'sans'
    plt.rcParams["font.size"] = 11
    plt.rcParams["text.color"] = "black"

    if plt.fignum_exists(figtitle):  # MANDATORY. If a new simulation is launched the previous window must be closed to avoid error.
        plt.figure(figtitle)                            # activate the figure of the simulation.
        fig = plt.gcf()
        wm = plt.get_current_fig_manager()              # Get the window position.
        wgeometry = wm.window.geometry()
        wgeometry = wgeometry[wgeometry.index("+"):]    # Keep the position only and remove the size. NECESSARY for appropriate figure scaling.
        figsize = fig.get_size_inches()                 # Get the current figure size.
        plt.close(fig)                                  # Close simulation figure.
    else:                                               # Otherwise set the default figure parameters: position and size.
        wgeometry = "+400+150"
        figsize = (2 * figheight, figheight)

    fig = plt.figure(figtitle, figsize=figsize)         # Create a simulation figure.
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry(wgeometry)

    # Get colors and types.
    cells = list(cellcolors.keys())
    types = {category:i for i,(category,*_) in enumerate(cells)}  # types is a dictionary {category:position in cells}.
    colors = [cellcolors[cell] for cell in cells]

    # Axe of CA + initialization of the CA display.
    X0 = 0.02  # Left bottom position of the CA
    Y0 = 0.1
    axca = fig.add_axes([X0, Y0, 0.45, 0.9])
    axca.set_aspect('equal',adjustable='box', anchor='SW')

    # CA initialization where the cells are encoded by their index of type in types to properly suit with colors.
    ca_coded = np.array([[types[category] for category, *_ in row] for row in simulation[0]])
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
        ca_coded = np.array([[types[category] for category, *_ in row] for row in simulation[step]])
        caview.set_array(ca_coded)    # Update CA
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
        buttonlabeling(autorun.get())                   # Update the button label.
    _autorun_button.on_clicked(click_autorun_button)    # Event on autorun button.

    # Button save Animation
    ax_save_button = fig.add_axes([X0, Y0 - 0.05, 0.015, 0.03])  # ON/OFF button is on the left side of slider.
    SAVED_ICON = "$\u25BD$"  # triangle pointing down, empty shape
    SAVE_ICON = "$\u25BC$"   # triangle pointing down, filled shape
    _save_button = Button(ax_save_button, SAVE_ICON)
    saved = Switch(False)

    def click_save_button(_):
        global _save_button
        fps = 1500//delay  # estimation of the fps from the delay between frames to have the same time.
        writer = PillowWriter(fps=fps)
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
            msgput("Save in progress.")
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
_gridsize = 1           # CA grid size
_duration = 1           # Duration of the simulation
_selector = None        # Rectangular selector.
_cell = None            # Current cell for CA0 painting.
_radiobutton = None     # Radio button to select the current cell.
_radiotypes = None      # Radio button on types.
_ca0 = None             # CA0 = initial automaton

def GuiCA(
    local_fun,
    cellcolors: dict,
    figheight: int = 5,
    gridsize: int = 100,
    duration: int = 200,
    delay: int = 100
):
    """Graphical interface for cellular Automata.
        limited to 10 types at most.

    Args:
        local_fun (function): local update function of the CA.
        cellcolors (dict): {cell:color} colors associated to cells. Recall that a cell is a tuple (type, states ..)
        figheight (int, optional): height of the figure of the simulation view. Defaults to 5.
        gridsize (int, optional): maximal size of the CA grid. Defaults to 100.
        duration (int, optional): maximal duration of the simulation. Defaults to 200.
        delay (int, optional): delay in ms between two simulation steps. Defaults to 100.
    """
    assert all([isinstance(cell, tuple) for cell in cellcolors])   # Check that keys are tuples!
    assert all([isinstance(category, str) for category, *_ in cellcolors])  # check that the types are strings!
    assert len(cellcolors) <= 10  # limited to 10 parameters - see program to understand this limitation.
    assert figheight > 0
    assert gridsize > 0
    assert duration > 0
    assert delay > 0

    global _gridsize
    global _duration
    # Cell parameter
    TYPE = 0
    # Windows parameters
    GUIWIDTH = 1.5      # Width of the GUI
    GUISTEP = 0.15      # Extra width step associated to characters of type labels.
    GUIHEIGHT = 4       # Height of the GUI

    STRSTRIDE = 0.0025  # Width  of characters for measure between 0 and 1, font size = 8.
    STRHEIGHT = 0.05    # Height of characters for measure between 0 and 1.
    RBOFFSET = 0.08     # Offset for the radio button

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
    colors =cellcolors.values()
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
            (FRMLEFT, 0.2),
            FRMSIZE,
            0.65,
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
    # ! I don't find a better solution than setting i for types[i] by an explicit number. This limits the number of used parameters to 10.
    # ! [lambda val:weights.set(types[i],val) for i in range(n)] and [lambda val:weights.set(category,val) for category in types]  DOES NOT WORK (i = max for all buttons !?)
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

    # New CA button ===
    axnew_button =  plt.axes([FRMLEFT, 0.11, FRMSIZE, SLIDHEIGHT])
    new_button = Button(axnew_button, "NEW", color="silver", hovercolor="lightsalmon")
    def newclick(_): # Callback of new button.
            global _selector
            global _cell
            global _radiotypes
            global _ca0

            # Figure of initial CA generation
            figca0_title = "CA0"
            typewidth =  RBOFFSET + max(map(len, types)) * STRSTRIDE        # Width of the type box.
            typeheight = STRHEIGHT * len(types)                             # Height of the type box
            fullwidth = figheight + 0.35 +  0.15 * max(map(len, types))     # Full width radio button + CA

            if plt.fignum_exists(figca0_title):                 # if the figure already exists then use it.
                plt.figure(figca0_title)                        # activate the figure of CA 0
                figca0 = plt.gcf()
            else:                                               # create a new figure for the visualization of the initial CA = CA0.
                figsize = (fullwidth, figheight)
                figca0 = plt.figure(figca0_title, figsize = figsize)
                wm = plt.get_current_fig_manager()
                wm.window.wm_geometry("+400+150")

            # Radio button of categories
            _cell = list(cellcolors.keys())[0]  # Initialize the cell to key 0 for matching to default activated radio buttons.
            axradiobutton =  plt.axes([0.025, 0.025, typewidth, typeheight])
            axradiobutton.set_facecolor('whitesmoke')
            _radiotypes = RadioButtons(axradiobutton, tuple(types), activecolor='orangered', radio_props={'s':50}, active=0)
            for r in _radiotypes.labels:
                r.set_fontfamily("fantasy")
                r.set_fontsize(8)

            typecells = {category:(category,) + tuple(states) for category, *states in cellcolors}   # association of the categories to cells.
            def radioclick(label):
                global _cell
                _cell = typecells[label]
            _radiotypes.on_clicked(radioclick)

            # Cellular automata initialization
            axca0 = figca0.add_axes([0.04+typewidth, 0.025, 0.97*figheight/fullwidth, 0.97])
            axca0.set_aspect('equal',adjustable='box', anchor='C')  # Force the square shape of the CA display.
            _ca0 =  GenerateCA(_gridsize, cellcolors, weights.weights)
            ca0cat = np.array([[types.index(category) for category, *_ in row] for row in _ca0])
            ca0view = DrawCA(ca0cat,colors,axca0).collections[0]

            # Selector
            def onselect(eclick,erelease):
                global _cell
                xmin,xmax,ymin,ymax = (round(val) for val in _selector.extents)
                _ca0[ymin:ymax,xmin:xmax] = _cell               # Initialize the array area with the current default cell
                category, *_ = _cell
                ca0cat[ymin:ymax,xmin:xmax] = types.index(category)   # Initialize the array view area with the  index of the current category.
                ca0view.set_array(ca0cat)

            _selector = RectangleSelector(axca0,
                                        onselect,
                                        button=[1, 3],
                                        interactive=False,
                                        spancoords='data',
                                        use_data_coordinates=True,
                                        props=dict(facecolor='gray', edgecolor='black', linewidth=2, alpha=0.3, fill=True),
                                        )
            plt.show()
            return  # end of newclick function
    new_button.on_clicked(newclick)

    # Run Button ======
    axrun_button = plt.axes([FRMLEFT, 0.025, FRMSIZE, SLIDHEIGHT])
    run_button = Button(axrun_button, "RUN", color="silver", hovercolor="lightsalmon")

    def runclick(_):  # Run Button clicked
        global _gridsize
        global _duration
        global _animation
        global _ca0

        # check if at least a weight is different to 0
        if sum(weights.weights.values()) == 0:
            print("** CA WARNING: at least one weight must be different to 0.")
        else:
            if _ca0 is None:
                _ca0 =  GenerateCA(_gridsize, cellcolors, weights.weights)

            simulation = SimulateCA(_ca0, local_fun, duration=_duration)
            _animation = ShowSimulation(simulation, cellcolors, figheight=figheight, delay=delay)

    run_button.on_clicked(runclick)  # Event on button
    plt.show()
