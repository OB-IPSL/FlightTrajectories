README 

Flight trajectory optimisation

Author: Olivier Boucher & Ed Gryspeerdt

Licence: GNU General Public Licence v3.0

Date: February 2023

Files: 

PACKAGES

misc_geo.py                   python package with miscellaneous geographic functions
misc_time.py                  python package with miscellaneous time functions 
optimalrouting                python package for the Zermelo method

PREPROCESSING

screen_iagos_only.py          python script to screen IAGOS flights
screen_iagos_only.job         batch job to run the screening python script
prepare_hourly_era5.sh        script to prepare the ERA5 data

TEST INPUT FILE

IAGOS_timeseries_2019010510370802_L2_3.1.0.nc4

SCRIPT

optim_iagos_only.py           python script to optimize a trajectory or a set of trajectories
optim_iagos_only.job          batch job to run the flight optimization python script 
launch_optim_by_route.sh      bash script to launch jobs in bulk

POST-PROCESSING

optim_replot_iagos_only.py         python script to rerun and adjust the plots for a flight
optim_analyse_route_iagos_only.py  python script to analyse the results statistically
