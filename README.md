# Reinfocement Learning Amoebas Simulation

Only dependent on NumPy (and PyGame for the graphics),
this project demonstrates an evolving neural network.

- The amoebas receive readings from their virtual sensors (drawn as cute eyes) 
- The inputs are proccessed in their neural networks to detemrmine their motion's angle
- Amoebas who touch the walls, die
- The longest surviving Amoebas pass their DNAs and Neural structures to the next generation, with small mutations
- Eventually, the Amoebas learn how to avoid the walls

The package comes with two maze maps, and an option to make the walls spin around and stretch.

![screenshot of amoebas moving in maze](https://raw.githubusercontent.com/ozmerchavy2/reinforcement-learning-amoebas/main/amoeba-screenshot.png)

The package comes with pre-trained amoebas' DNA easy to load into the simulation (although it usually only takes them a couple of minutes to learn).
