# Thesis-MOOC-Data-Analysis

## Introduction
These scripts are made for analysing Computer Science MOOC data from the EdX platform. The data is collected from the courses "EX101x: Data Analysis EX101", "FP101x: Functional Programming", "ST1x: Automated Software Testing: Practical Skills for Java Developers", and "UnixTx: Unix Tools: Data, Software and Production Engineering". This repository contains scripts for creating an SQLite database from the data, as well as analysing and graphing the data. The actual data is found on the TU Delft servers under `\staff-umbrella\gdicsmoocs\`.

## Installation
The scripts are written in Python 3.12.1. To install the required packages, run the following command `pip install -r requirements.txt`.

### Creating the database
To create the database, run the following command `python scripts/db/create_and_populate_db.py`. This will create a database called `thesis_db` in the working directory `\staff-umbrella/gdicsmoocs/Working copy/scripts`/

### Run graphing scripts
To run the graphing scripts, run the command `python graphs/run_graph_scripts.py`.