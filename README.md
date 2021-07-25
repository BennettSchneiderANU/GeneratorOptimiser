# genOpt
Co-optimiser of electricity generation dispatch, particularly battery energy storage systems (bess).

Open the example file in example/bessOpt_example.py for a demo of how to use this package.

This package has 5 sub-packages:
1. classes -> contains object constructors for Network (and NEM subclass) objects and Gen (and BESS subclass) objects and associated methods.
2. analysis_functions -> Contains all the helper functions to run analysis on which the class methods rely heavily. This includes optimised dispatch programs.
3. plotting_functions -> Contains functions for plotting results in plotly offline.
4. error_handling -> A basic set of helper functions to handle errors nicely.
5. genOpt -> Master package to combine all sub-packages under a single umbrella for easy import.

To use these packages clone the repo, add its location to your python path and import using:
import genOpt as go

All functions will then be available under "go", e.g.
go.BESS for the BESS object constructor or;
go.horizonDispatch for the horizon dispatch function.







