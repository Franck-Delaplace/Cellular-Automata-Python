Library of cellular automaton written in Python with examples including the game of life.
This library is used for my Master course at Paris-Saclay University, Evry.
Please feel free to use it for educational purposes or other.

<code>celllularautomata</code> is the name of the library. The other programs are examples for using the library.
See game-of-life for a tutorial to use the library. See game-of-life or forest-fire examples for a demo.

The simulation can be controlled using the GUI. To adjust the weight of each cell type, use the slider provided.  The GUI includes two buttons: NEW and RUN. Clicking NEW generates a new initial state for the CA and opens a window where you can modify the cells graphically using a region selector. Clicking RUN starts the simulation.

Clicking NEW generates a new initial state and opens a window where you can modify the cells graphically. The configuration of this window will be the initial state of the CA as long as it remains open.

To display the simulation, simply click on RUN, and a new window will open. You can then save the simulation as a GIF. The waiting delay depends on the number of steps requested, as the simulation is pre-calculated.

To install the project, go in the directory of the CA library type: 
<ul>
<li> <code>pip install .</code>  </li>
<li> or,  <code> python -m pip install .</code></li>
</ul>
