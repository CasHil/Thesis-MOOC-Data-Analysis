# Thesis-MOOC-Data-Analysis

This repository contains the code for my computer science master thesis: [Debugging the Divide: Exploring Men's and Women's Motivations and Engagement in Computer Science MOOCs](https://repository.tudelft.nl/record/uuid:f4aceeec-5947-4578-834c-4bb43288c91a).

The goal of this research was to answer three research questions:
RQ1: What are the differences in reasons for enrolment between men and women in introductory computer science MOOCs?
RQ2: What are the differences in behavioural engagement between men and women in introductory computer science MOOCs?
RQ3: How do reasons for enrolment influence behavioural engagement among men and women in introductory computer science MOOCs?

## Introduction

These scripts are made for analysing Computer Science MOOC data from the EdX platform. The data is collected from the following courses:
| Course ID | Course name |
| ----------- | ----------- |
| EX101x | Data Analysis |
| FP101x | Functional Programming |
| ST1x | Automated Software Testing: Practical Skills for Java Developers |
| UnixTx | Unix Tools: Data, Software and Production Engineering |

This repository contains scripts for creating an SQLite database from the MOOC (meta)data, as well as scripts for the performing the experiments as described in my thesis.

This thesis makes use of data which can be requested from the TU Delft Extension School, but any [EdX data delivery](https://edx.readthedocs.io/projects/devdata/en/latest/using/package.html) should work. An EdX data delivery should look like this:

| Folder Name        |
| ------------------ |
| EX101x_2T2018_run6 |
| EX101x_3T2014_run4 |
| EX101x_3T2015_run2 |
| EX101x_3T2016_run4 |
| FP101x_3T2015_run2 |
| STIx_3T2018_run4   |
| STIx_3T2019_run2   |
| STIx_3T2020_run3   |
| STIx_3T2021_run4   |
| STIx_3T2022_run5   |
| UnixTx_1T2020_run1 |
| UnixTx_1T2022_run4 |
| UnixTx_2T2021_run3 |
| UnixTx_3T2020_run4 |

If we take the course run EX101x_2T2018_run6 as an example, it should contain the following files:
| File Name |
|----------------------------------------------------------------------|
| delftx-edx-events-2018-12-13.log.gz |
| delftx-edx-events-2018-12-14.log.gz |
| More events files... |
| DelftX-EX101x-2T2018-auth_userprofile-prod-analytics.sql |
| DelftX-EX101x-2T2018-certificates_generatedcertificate-prod-analytics.sql |
| DelftX-EX101x-2T2018-course_structure-prod-analytics.json |
| DelftX-EX101x-2T2018-course-prod-analytics.xml.tar.gz |
| DelftX-EX101x-2T2018-grades_persistentcoursegrade-prod-analytics.sql |
| DelftX-EX101x-2T2018-student_courseaccessrole-prod-analytics.sql |
| DelftX-EX101x-2T2018-student_courseenrollment-prod-analytics.sql |
| mid_survey_results_EX101x_2T2018.txt |
| post_survey_results_EX101x_2T2018.txt |
| pre_survey_results_EX101x_2T2018.txt |

## Installation

The scripts are written in Python 3.12.1. To install the required packages, run `pip install -r requirements.txt`.

### Creating the demographic database

First, change the `.env example` file to use the EdX data delivery folder as a base folder.
To create the database, run the following command: `python scripts/setup.py`. This will create a folder called `scripts/mooc_db` in your base folder with a database called `mooc_db`.

### Run demographic graph scripts

To run the demographic graph scripts, run the command `python graphs/run_graph_scripts.py`.

### Creating the interaction database

To answer my research questions, a database of learner interactions with the MOOCs was required. I made a fork of [ELAT-Node](https://github.com/mvallet91/ELAT-Node) (thanks Manuel!) to do this.

Make sure you have `node >=11.14.0` installed. Then run `npm i` and `node scripts/engagement/behavioural/learning_paths/ELAT-Node/index.js`. Note: this takes quite a while.

### Classifying learner engagement

One of the most important scripts for my thesis is about learning trajectories in MOOCs, inspired by René F Kizilcec, Chris Piech, and Emily Schneider. "Deconstructing disengagement: analyzing learner subpopulations in massive open online courses". In: Proceedings of the third international conference on learning analytics and knowledge. 2013, pp. 170-179.

This script can be found at `scripts/engagement/behavioural/learning_trajectories/k_means_clustering.py`. Further explanations can be found at the top of that script as comments.

### Final notes

There are also quite a few ideas I had that didn't end up working out, like different ways to measure learner engagement (cognitive engagement, for example). I have mentioned the most important scripts I used for my thesis. Some of the remaining scripts could serve as ideas for future research. Other scripts serve as utilities for transforming data from one form into another, for example.

Questions? I am no longer available through my TU Delft mail, but you can contact me through my personal email: casperhildebrand (at) gmail.com.
