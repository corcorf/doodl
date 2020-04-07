# doodl
Markov Chain-based simulation of customer movements in a fictional supermarket.

## Features
* Customer implemented as a generator class, allowing the next customer state to be generated as needed.
  * Different customer sub-classes to be created to simulated different behaviours at different times of day. 
* Supermarket implemented as a class containing a variable number of customers.
* Transition matrices handled in numpy and pandas.
* Command line interface allowing number of checkouts, checkout rates and other options to be set.
* Animation of simulated customer locations over time using OpenCV2.

## Images
![Example animation](images/doodl.gif)

## Usage
To run a simulation with the default settings and display the resulting animation:

```python supermarket.py```

The simulation and display settings can be controlled with a command-line interface. 

To view the available options:

```python supermarket.py --help```
