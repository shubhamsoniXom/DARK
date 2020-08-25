![Data Science & Optimization](https://mysite.na.xom.com/personal/upstreamaccts_shusoni/Documents/Shared%20with%20Everyone/logo.png)

ML tech Stack
==============================
In the Spirit of the quote :
> “*Interdependence is greater than independence*” 
>    –Stephen Covey, in “The 7 Habits of Highly Effective People”

This repo targets to ASK & PROVIDE most commonly and repititively used methods in day to day
Data Science activities.

We often practice these two elements in our daily job:-

**R** : Read, Research & some times Reach-out to others

**I** : Implement, Improvise or Innovate 

let's bring in another element which puts our minds to Peace when it comes to re-development 
of scripts from scratch for every project/PoC.

**P** : Promote Collaboration, Practice sharing, Peer-to-Peer reviews and most importantly 
        Provide faster results to clients.

Summary:

This repo is built upon cookiecutter data-science template and can be installed as any other 
python package. The repo aims to cover best practices and variety of methods for each step
in Data Science work flow.

Typical DS workflow:

Data Collection > Data Preparation > Exploratory Data analysis > Modeling > Testing > Deployment

Repo contents:

**Tableau:** 

    1.  Run python code in Tableau - Refer to python_in_tableau notebook.
        
        
**Data Extraction Utilities:**

    1. Extract various formats of Dates from string using Regex - <notebook name>
    2. Extract tables from pdf using various python packages
    3. Download data (excel or pickle files) from Azure blob storage using python
    
    

more good stuff coming soon!!!

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
