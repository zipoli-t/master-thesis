# Data Preprocessing Scripts

This directory contains the scripts required to transform raw data into pandas-readable CSV files. The pipeline preserves the authors' original transformation logic (e.g., log-levels, first differences) to ensure data stationarity.

## Third-party software attribution
The sub-directory `fred-database_code/` contains the official FRED-MD MATLAB Package developed by:
* **Authors**: Michael W. McCracken and Serena Ng
* **Original publication**: McCracken, M.W., Ng, S., 2015; FRED-MD: A Monthly Database for Macroeconomic Research, Federal Reserve Bank of St. Louis Working Paper 2015-012. URL: https://doi.org/10.20955/wp.2015.012
* **Code source**: https://www.stlouisfed.org/research/economists/mccracken/fred-databases

## Modifications for this project
The following technical adaptations were implemented for integration:

1. Automated directory discovery for relative pathing.
2. Deactivation of outlier removal and factor extraction (uncomment relevant lines in the script if these features are required).
3. Automated export to CSV format.

## Usage
To run the preprocessing pipeline:
1. Ensure the raw .csv data files are placed in `/data/raw/`.
2. Run `fredfactors.m` in MATLAB.
3. The processed output will be automatically saved to `/data/processed/`.