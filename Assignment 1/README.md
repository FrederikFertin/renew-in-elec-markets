# Assignment 1
This repository contains the code and data for assignment 1.
The pdf file Assignment_1_report.pdf is the answer report.

## Bus system and Data used
The wind data used in taken from the website: "https://sites.google.com/site/datasmopf/data-format". The zone chosen is zone 8. The file has been imported as a csv and renamed: "wind_profiles.csv".

## Subfolders
The folder contains a folder for the inpt data, the plots generated, and the assignment description.

## Scripts
Most scripts, except network_plots.py and Step_2.py, contain a main, which should be run to obtain results for the given step in the assignment. All scripts define functions and/or classes.

### network_plots.py
This script contains functions to build and plot a pandapower electrical grid model for visualization of results.

### Step_1_2.py
This script contains the basic class Network, which is inherited by all other classes and contains the input data about the network of the task. It also contains the class Economic Dispatch, which is used to clear the day ahead market with and without intertemporal constraints.

### Step_2.py
This script contains the functions required for the intertemporal economic dispatch to run - and also contains the general balance equation constraints.

### Step_4.py
This script contains the class for a nodal and/or zonal market clearing called NodalMarketClearing.

### Step_5.py
This script contains a BalancingMarket class, which can be used to clear the balacning market after the day ahead market has been cleared.

### Step_6.py
This script contains a ReserveAndDispatch class, clearing the reserve market and then the day-ahead market.


