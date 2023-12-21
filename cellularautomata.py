
# * CELLULAR AUTOMATA LIBRARY
# * Author: Franck - Dec. 2023
# * MASTER TUTORIAL
# * Paris Saclay University

import matplotlib as mpl
import matplotlib.pyplot as plt
from random import choices
import numpy as np                                                                              # type: ignore
from matplotlib.animation import FuncAnimation, PillowWriter                                    # type: ignore
import matplotlib.colors as color                                                               # type: ignore
import seaborn as sns                                                                           # type: ignore
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons, RectangleSelector    # type: ignore
from matplotlib.patches import Rectangle                                                        # type: ignore
from tqdm import tqdm


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
    """Generate n*n 2D cellular automaton randomly.

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
    try:    # Generate the cells randomly
        randca = choices(cells, weights=weights_, k=n * n)
    except ValueError:
        print("** CA ERROR: At least one weight must be greater to 0.")
        exit()

    return np.array([[randca[i + n * j] for i in range(n)] for j in range(n)])  # Reshape to get a 2D array


def Moore(r: int) -> list[tuple[int, int]]:
    """Compute the Moore neighborhood of radius r.

    Args:
        r (int): radius.

    Returns:
        list[tuple[int]]: Moore neighborhood.
    """
    moore = [(x, y) for x in range(-r, r+1) for y in range(-r, r+1)]
    moore.remove((0, 0))
    return moore


def VonNeumann(r: int) -> list[tuple[int, int]]:
    """Compute the Von Neumann neighborhood of radius r.

    Args:
        r (int): radius.

    Returns:
        list[tuple[int]]: Von Neumann neighborhood.
    """
    vonneumann = [(x, 0) for x in range(-r, r+1)]+[(0, y) for y in range(-r, r+1)]
    vonneumann.remove((0, 0))
    vonneumann.remove((0, 0))
    return vonneumann


def DrawCA(cellautomaton: np.ndarray, colors: list, ax):
    """Draw a 2D cellular automaton

    Args:
        cellautomaton (np.array): Cellular Automata
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


def SimulateCA(cellautomaton0: np.ndarray, f, neighborhood=Moore(1), duration: int = 100) -> list:
    """Compute a simulation of a cellular automaton.

    Args:
        cellautomaton0 (np.ndarray): initial cellular automata
        f (fun): local update function
        neighborhood (list[tuple], optional): cell neighborhood as list of 2D displacements (use MOORE or VONNEUMANN). Default MOORE.
        duration (int, optional): total number of steps. Default 100.

    Returns:
        list: Simulation trace corresponding to a list of cellular automata.
    """
    assert duration >= 0

    def ca_step(cellautomaton: np.ndarray, fun) -> np.ndarray:  # Compute 1 CA step.
        # Displacement of the Moore neighborhood
        n = len(cellautomaton)
        mooreshift = np.array([np.roll(cellautomaton, dis, axis=(0, 1)) for dis in neighborhood])  # Copies of CA cyclically shifted according to Moore's neighborhood
        neighborsgrid = list(np.transpose(mooreshift, axes=(1, 2, 0, 3)))                   # Transposition to obtain a 2D array of neighbor lists
        canew = np.array(
                [[fun(cellautomaton[i][j], neighborsgrid[i][j]) for j in range(n)] for i in range(n)]
                )  # apply the local evolution function
        return canew

    simulation = [cellautomaton0]
    try:
        for i in tqdm(range(duration), desc="CA Step", ascii=False, bar_format="{l_bar}{bar:65} {r_bar}", colour='#3d8c40'):  # With progress bar.
            simulation.append(ca_step(simulation[i], f))
    except ValueError:
        print("** CA ERROR: Invalid cell format encountered. A condition on cell is probably missing in the local function.")
        exit()

    return simulation


# Switch for managing the Boolean flags.
class Switch:
    """Boolean value toggling for switch control."""
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
    figtitle = "CELLULAR AUTOMATON - FD MASTER COURSE"  # Simulation title.

    # Font style for all texts in the simulation window.
    mpl.rcParams["font.family"] = "fantasy"
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["text.color"] = "black"

    if plt.fignum_exists(figtitle):                     # Activate and identify the figure if it already exists
        plt.figure(figtitle)
        fig = plt.gcf()
        wm = plt.get_current_fig_manager()              # Get the window geometry and figure size
        wgeometry = wm.window.geometry()
        wgeometry = wgeometry[wgeometry.index("+"):]    # Keep the position only and remove the size. NECESSARY for appropriate figure scaling.
        figsize = fig.get_size_inches()
        plt.close(fig)                                  # ! Check if really needed, seems to be to avoid error.
    else:                                               # Otherwise set the default figure parameters: position and size.
        wgeometry = "+450+150"
        figsize = (2 * figheight, figheight)

    fig = plt.figure(figtitle, figsize=figsize)         # Create the simulation figure.
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry(wgeometry)

    # Get colors and types of the cells
    cells = list(cellcolors.keys())
    types = {category: i for i, (category, *_) in enumerate(cells)}  # Types is a dictionary {category:position in cells}.
    colors = [cellcolors[cell] for cell in cells]

    # Axis of CA + initialization of the CA display.
    X0 = 0.02  # Left bottom position of the CA
    Y0 = 0.1
    axca = fig.add_axes((X0, Y0, 0.45, 0.9))
    axca.set_aspect('equal', adjustable='box', anchor=(0, 1))

    # The cells are encoded by the index of the category in types to suit with array format of heatmap in DrawCA.
    ca_heatmap = np.array([[types[category] for category, *_ in row] for row in simulation[0]])
    caview = DrawCA(ca_heatmap, colors, axca).collections[0]

    # Axe of curves
    CHEIGHT: float = 0.87  # Height of the curve axe.
    axcurve = fig.add_axes((X0 + 0.52, Y0, 0.44, CHEIGHT))
    axcurve.set_xlim(0, n)
    axcurve.set_ylim(0, len(simulation[0]) ** 2)
    axcurve.grid(linestyle="--")

    # Initialize the count curves.
    typescount = {  # Dictionary keeping the count of the different cell types.
        category: [sum([CountType(row, category) for row in ca]) for ca in simulation]
        for category in types}

    visible_curves = [thecolor != "white" for thecolor in colors]  # All the curves are visible but those drawn in white color.
    curves = {                                               # The curves are collected to a dictionary {type: counting curve}.
        category: axcurve.plot(
            [0],
            typescount[category][0],
            color=colors[i] if colors[i] != "white" else "lightgray",  # The white color is transformed into a light gray.
            linewidth=2.5,
            marker=" ",
            visible=visible_curves[i],
        )[0]
        for i, category in enumerate(types)}

    # || Check box button characterization for curves
    chxboxheight = len(types) * 0.05                    # Check box height which depends on the number of categories.
    chxboxwidth = 0.05 + max(map(len, types)) * 0.006   # Check box width which depends  on the maximal string length of the categories.
    axcurves = fig.add_axes(
        (X0 + 0.52, Y0 + CHEIGHT - chxboxheight, chxboxwidth, chxboxheight))    # The checkboxes are located in the upper left of the curve graphics.

    _curve_button = CheckButtons(axcurves, types, visible_curves)

    def chxboxupdate(category: str) -> bool:  # Update the checkboxes.
        return curves[category].set_visible(not curves[category].get_visible())  # Toggle the visibility of curve.
    _curve_button.on_clicked(chxboxupdate)

    # || Slider definition to control the progression of the simulation
    axslider = fig.add_axes((X0 + 0.04, Y0 - 0.07, 0.412, 0.07))    # The slider is located below the cellular automaton display.
    slider = Slider(axslider, "", 0, n - 1, valstep=1, valinit=0, facecolor="gray", valfmt="%3d")

    xrange = np.arange(0, n, 1, dtype=int)

    def updateslider(step):  # Update of the slider.
        ca_coded = np.array([[types[category] for category, *_ in row] for row in simulation[step]])
        caview.set_array(ca_coded)          # Update CA
        for category in types:              # Update type count curves
            curves[category].set_data(xrange[:step], typescount[category][:step])
        return
    slider.on_changed(updateslider)         # Event on slider.

    # || ON/OFF autorun Button.
    ax_autorun_button = fig.add_axes((X0+0.02, Y0 - 0.05, 0.015, 0.03))  # ON/OFF button is on the left side of slider.
    _autorun_button = Button(ax_autorun_button, " ")

    # Button labeling to indicate autorun status.
    OFF_ICON: str = "$\u25a0$"  # Square
    ON_ICON: str = "$\u25B6$"   # Right triangle

    def buttonlabeling(state: bool):  # Set the label ON/OFF to the button w.r.t. to a Boolean state.
        _autorun_button.label.set_text({False: ON_ICON, True: OFF_ICON}[state])

    buttonlabeling(autorun.get())  # Initialize button label from the initial autorun state.

    def click_autorun_button(_):  # autorun button call back
        global _autorun_button
        autorun.switch()                                # Switch the autorun.
        buttonlabeling(autorun.get())                   # Update the button label.
    _autorun_button.on_clicked(click_autorun_button)    # Event on autorun button.

    # || Button to save Animation
    ax_save_button = fig.add_axes((X0, Y0 - 0.05, 0.015, 0.03))  # ON/OFF button is on the left side of slider.
    SAVED_ICON = "$\u25BD$"  # Triangle pointing down, empty shape
    SAVE_ICON = "$\u25BC$"   # Triangle pointing down, filled shape
    _save_button = Button(ax_save_button, SAVE_ICON)
    saved = Switch(False)

    def click_save_button(_):
        global _save_button
        fps = 1000//delay  # Estimation of the fps from the delay between frames to have the same time.
        writer = PillowWriter(fps=fps)
        _animation.save("CA-SIMULATION.gif", writer=writer)
        msgput("Save completed!")
        saved.set(True)
        _save_button.label.set_text(SAVED_ICON)
    _save_button.on_clicked(click_save_button)  # Event on save button.

    # || Tooltips handler
    axmsg = fig.add_axes((X0, Y0 - 0.09, 0.45, 0.03), facecolor="gainsboro")   # The message box is below the slider

    def msgclear():         # Clear the message box.
        axmsg.cla()
        axmsg.set_xticks([])
        axmsg.set_yticks([])

    def msgput(msg: str):   # Print a message in the message box.
        msgclear()
        axmsg.text(0.01, 0.2, msg, fontsize=8, fontfamily='serif', fontstyle='italic')

    # Handling events: move + click on the axes.
    def hover(event):
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

    def onclick(event):
        if ax_save_button.contains(event)[0]:
            msgput("Save in progress.")
        elif ax_autorun_button.contains(event)[0]:
            msgput("Simulation switched "+("OFF, scroll the slider." if autorun.get() else "ON."))
        else:
            pass

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", onclick)

    msgclear()  # Initially clear the message box.

    # || Display simulation
    def updateanimation(_):         # Update from animation.
        if autorun.get():           # The update is conditional on the state of autorun.
            step = (slider.val + 1) % slider.valmax
            slider.set_val(step)    # Updating slider value also triggers the updateslider function

    _animation = FuncAnimation(fig, updateanimation, interval=delay, save_count=n)  # Run animation.
    fig.show()
    return _animation


# Class to manage weights for random definition of the CA grid.
class Weights:
    """Weights = dictionary associating cells to their weights which are floats between 0 and 1"""
    weights: dict = {}

    def __init__(self, types: list, value: float = 0.0):
        self.weights = {category: value for category in types}

    def set(self, category, val):  # Set the weight of a category.
        self.weights[category] = val

    def get(self, category):            # Get the weight of a state.
        return self.weights[category]


# Global variables used to pass arguments in sliders and buttons in GuiCA
_gridsize = 1               # CA grid.
_duration = 1               # Duration of the simulation.
_cell = None                # Current cell used to paint the selected area with this cell.
_ca0 = None                 # CA0 = initial automaton.
_neighborfun = Moore        # Function qualifying the neighborhood shape (Moore or VonNeumann).
_radius = 1                 # neighborhood radius
# Widgets
_radiotypes = None          # Radio button of NEW window.
_selector = None            # Cell selector of NEW window.
_neighbors_radio = None     # neighborhood radio button


def GuiCA(
    local_fun,
    cellcolors: dict,
    figheight: int = 5,
    gridsize: int = 100,
    duration: int = 200,
    delay: int = 100
):
    """Graphical interface for cellular Automata.
        The number of different cell types is limited to 10 at most.

    Args:
        local_fun (function): local update function of the CA.
        cellcolors (dict): {cell:color} colors associated to cells. Recall that a cell is a tuple (type, states ...)
        figheight (int, optional): height of the figure of the simulation view. Defaults to 5.
        gridsize (int, optional): maximal size of the CA grid. Defaults to 100.
        duration (int, optional): maximal duration of the simulation. Defaults to 200.
        delay (int, optional): delay in ms between two simulation steps. Defaults to 100.
    """
    assert all([isinstance(cell, tuple) for cell in cellcolors])   # Check that keys are tuples.
    assert all([isinstance(category, str) for category, *_ in cellcolors])  # check that the types are strings.
    assert len(cellcolors) <= 10  # limit to 10 parameters - see program to understand this limitation.
    assert figheight > 0
    assert gridsize > 0
    assert duration > 0
    assert delay > 0

    global _gridsize
    global _duration
    global _neighbors_radio

    # Windows parameters
    GUIWIDTH: float = 1.5                   # Minimal width of the GUI figure.
    GUISTRSTRIDE: float = 0.12              # Stride associated to character used for figure width definition.
    GUIHEIGHT: float = 5.5                  # Height of the GUI figure. This value must be adapted to the number of types.
    # Rectangles
    FRMLEFT: float = 0.07                   # Frame left position
    FRMHEIGHT: float = 0.86                 # Frame size
    FRMEDGECOLOR: str = "darkgray"          # Frame color
    # Button & Slider parameters
    MAXRADIUS: int = 3                      # Maximal neighborhood radius size
    SLIDLEFT: float = 0.35                  # Left position of sliders.
    SLIDSIZE: float = 0.4                   # Size of sliders.
    WIDGHEIGHT: float = 0.07                # Height of sliders and buttons.
    SLIDSTART: float = 0.67                 # Vertical start position for weight sliders.
    SLIDDIST: float = 0.05                  # Distance between two weight sliders.
    SLIDCOLOR: str = "gray"                 # Slider color bar.
    # Radio button parameters
    RADIOFFSET: float = 0.015               # Minimal incompressible distance in a radio button.
    RADIOSTRSTRIDE: float = 0.010           # Stride for characters in radio button.
    RADIOSTRIDE: float = 0.01               # Stride between two radio buttons.
    BUTTONCOLOR: str = "silver"             # Standard color of buttons.
    HOVERCOLOR: str = "lightsalmon"         # Hover color of buttons.
    UNSELECTCOLOR: str = "lemonchiffon"     # Color of the radio button when it is unselected.
    SELECTCOLOR: str = 'gold'               # Color of the radio button when it is selected.

    # Initialization of the main variables
    _gridsize = gridsize // 2
    _duration = duration // 2
    cells = list(cellcolors.keys())
    types = [category for category, *_ in cells]  # Get all types of cells
    colors = cellcolors.values()
    n = len(types)
    weights = Weights(types, 0.5)                 # Create weights from types.

    # Initialization of the figure
    mpl.rcParams["toolbar"] = "None"    # No toolbars on GUI figure.
    mpl.rcParams["font.family"] = "sans"
    mpl.rcParams["font.size"] = 8

    figui = plt.figure(figsize=(GUIWIDTH + max(map(len, types)) * GUISTRSTRIDE, GUIHEIGHT), num="GUI")
    ax = figui.add_axes((0, 0, 1, 1))
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("+50+100")

    # || Slider of neighborhood radius ===
    axradius_slider = figui.add_axes((SLIDLEFT, 0.93, SLIDSIZE, WIDGHEIGHT))
    radius_slider = Slider(
        axradius_slider,
        "Radius  ",
        1,
        MAXRADIUS,
        valstep=1,
        valinit=1,
        facecolor=SLIDCOLOR,
        valfmt="%3d",
        )

    def update_radius_slider(val):  # Event of radius slider
        global _radius
        _radius = val
    radius_slider.on_changed(update_radius_slider)

    # || Neighborhood radio button ===
    axneighbors_radio = figui.add_axes((FRMLEFT, 0.86, FRMHEIGHT, WIDGHEIGHT))
    for pos in ['left', 'bottom', 'right', 'top']:
        axneighbors_radio.spines[pos].set_color(FRMEDGECOLOR)
        axneighbors_radio.spines[pos].set_linewidth(2)

    # Dictionary for neighborhood selection. The values are neighborhood design function.
    neighborhood = {"Moore": Moore, "Von Neumann": VonNeumann}
    _neighbors_radio = RadioButtons(axneighbors_radio,
                                    list(neighborhood.keys()),
                                    activecolor=BUTTONCOLOR,
                                    radio_props={'s': 30},)

    def neighborsclick(label):  # Radio neighborhood callback.
        global _neighborfun
        _neighborfun = neighborhood[label]
    _neighbors_radio.on_clicked(neighborsclick)

    # || Grid size slider ======
    axsize_slider = figui.add_axes((SLIDLEFT, 0.79, SLIDSIZE, WIDGHEIGHT))
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

    def update_slider_size(val):
        global _gridsize
        _gridsize = val
    size_slider.on_changed(update_slider_size)  # Event on size slider

    # || Duration/Time sliders ======
    axduration_slider = figui.add_axes((SLIDLEFT, 0.74, SLIDSIZE, WIDGHEIGHT))
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

    # || Weights  sliders ======
    # Rectangle
    ax.add_patch(
        Rectangle(
            (FRMLEFT, 0.2),
            FRMHEIGHT,
            0.54,
            facecolor="whitesmoke",
            edgecolor=FRMEDGECOLOR,
            linewidth=2,
        )
    )

    # Weight sliders definition
    weight_sliders = [
        Slider(
            figui.add_axes((SLIDLEFT, SLIDSTART - i * SLIDDIST, SLIDSIZE, WIDGHEIGHT), facecolor=SLIDCOLOR,),
            str(types[i]) + "  ",
            0.0,
            1.0,
            valstep=0.005,
            valinit=0.5,
            facecolor=SLIDCOLOR,
            valfmt="%1.3f",
        ) for i in range(n)
        ]

    # All possible updates for 10 weight sliders at most.
    # ! I don't find a better solution than setting i for types[i] by an explicit number.
    # [lambda val:weights.set(types[i],val) for i in range(n)] and [lambda val:weights.set(category,val) for category in types]  DOES NOT WORK (i = max for all buttons !?)
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

    # || New CA button ===
    axnew_button = figui.add_axes((FRMLEFT, 0.11, FRMHEIGHT, WIDGHEIGHT))
    new_button = Button(axnew_button, "NEW", color=BUTTONCOLOR, hovercolor=HOVERCOLOR)

    def newclick(_):  # Callback of NEW button.
        global _cell
        global _ca0
        global _radiotypes
        global _selector

        # Figure of initial CA generation
        figca0_title = "CA0"

        if plt.fignum_exists(figca0_title):                 # If the figure already exists then use it.
            plt.figure(figca0_title)                        # Activate the figure of CA0.
            figca0 = plt.gcf()
        else:                                               # Otherwise create a new figure for the visualization of the initial CA = CA0.
            figsize = (figheight, figheight + 0.5)
            figca0 = plt.figure(figca0_title, figsize=figsize)
            wm = plt.get_current_fig_manager()
            wm.window.wm_geometry("+450+150")

        # Cellular automata initialization - generation of the initial CA (CA0).
        axca0 = figca0.add_axes((0.01, 0.0, 0.98, 0.98))
        plt.subplots_adjust(bottom=0.1)
        axca0.set_aspect('equal', anchor=(0.5, 1.0))  # The CA0 drawing is anchored in the middle top.

        _ca0 = GenerateCA(_gridsize, cellcolors, weights.weights)
        ca0code = np.array([[types.index(category) for category, *_ in row] for row in _ca0])
        ca0view = DrawCA(ca0code, list(colors), axca0).collections[0]

        # Radio button of categories
        radiofullwidth = n * (max(map(len, types)) * RADIOSTRSTRIDE + RADIOSTRIDE + RADIOFFSET)     # Full width of the button bar.
        radiospacing = radiofullwidth/n                                                             # Distance between 2 radio buttons.

        axradio = [figca0.add_axes(((0.5 - radiofullwidth/2) + i * radiospacing + RADIOSTRIDE, 0.02, radiospacing - RADIOSTRIDE, WIDGHEIGHT/1.5))
                   for i in range(n)]

        _radiotypes = [Button(axradio[i], category, color=UNSELECTCOLOR, hovercolor=HOVERCOLOR) for i, category in enumerate(types)]
        for rb in _radiotypes:              # Set style of the radio button labels.
            rb.label.set_fontfamily("fantasy")
            rb.label.set_fontsize(8)

        # Initialization of the radio button bar
        _radiotypes[0].color = SELECTCOLOR  # The first button is the default button. Assign to the color 'selected'
        _cell = list(cellcolors.keys())[0]  # the default cell is the first one in cellcolors.

        def radioclick(index: int):  # Radio click call back with the index of the type as input.
            global _cell
            for rb in _radiotypes:                  # Unselect all radio buttons.
                rb.color = UNSELECTCOLOR
            _radiotypes[index].color = SELECTCOLOR  # Select the radio button corresponding to index.
            _cell = cells[index]

        radioclickfun = [  # Manually pre-defined 10 on-click radio-button functions. the problem is the same as weight sliders.
            lambda _: radioclick(0),
            lambda _: radioclick(1),
            lambda _: radioclick(2),
            lambda _: radioclick(3),
            lambda _: radioclick(4),
            lambda _: radioclick(5),
            lambda _: radioclick(6),
            lambda _: radioclick(7),
            lambda _: radioclick(8),
            lambda _: radioclick(9)]
        for i in range(n):
            _radiotypes[i].on_clicked(radioclickfun[i])

        # || Region Selector
        def onselect(eclick, erelease):
            global _cell
            xmin, xmax, ymin, ymax = (round(val) for val in _selector.extents)
            _ca0[ymin:ymax, xmin:xmax] = _cell   # Fill the selected array area with the default cell.
            category, *_ = _cell
            ca0code[ymin:ymax, xmin:xmax] = types.index(category)   # Fill the array view area with the index of _cell category.
            ca0view.set_array(ca0code)

        _selector = RectangleSelector(axca0,
                                      onselect,
                                      button=[1, 3],
                                      interactive=False,
                                      spancoords='data',
                                      use_data_coordinates=True,
                                      props=dict(facecolor='red', edgecolor='black', linewidth=2, alpha=0.3, fill=True),
                                      )

        figca0.show()
        return  # * end of newclick function
    new_button.on_clicked(newclick)

    # || Run Button ======
    axrun_button = figui.add_axes((FRMLEFT, 0.025, FRMHEIGHT, WIDGHEIGHT))
    run_button = Button(axrun_button, "RUN", color=BUTTONCOLOR, hovercolor=HOVERCOLOR)

    def runclick(_):  # Run Button clicked
        global _gridsize
        global _duration
        global _animation
        global _ca0
        global _neighborfun
        global _radius

        if _ca0 is None:  # When CA0 is not yet generated.
            _ca0 = GenerateCA(_gridsize, cellcolors, weights.weights)

        simulation = SimulateCA(_ca0, local_fun, neighborhood=_neighborfun(_radius), duration=_duration)
        _animation = ShowSimulation(simulation, cellcolors, figheight=figheight, delay=delay)
    run_button.on_clicked(runclick)  # Event on button

    plt.show(block=True)

    return  # End of GuiCA
