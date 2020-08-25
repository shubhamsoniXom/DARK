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

    1. Extract various formats of Dates from string using Regex - Extract_Dates_using_Regex.ipynb
    2. Extract tables from pdf using various python packages - todo
    3. Download data (excel or pickle files) from Azure blob storage using python - todo

**Dash Application Development Example**

    1. Wine classification example (Used in Knowledge sharing session by Sharad)
        To run this demo app, download this repo to your local navigate to ml_tech_stack (home) folder
        and after installing all required packages (do this in a virtual environment), from your terminal
        run <python app\app_wshp_final.py> this should launch an application on your localhost; you should see
        following output in your terminal 
        <Dash is running on http://127.0.0.1:8050/

        Warning: This is a development server. Do not use app.run_server
        in production, use a production WSGI server like gunicorn instead.

        * Serving Flask app "app_wshp_final" (lazy loading)
        * Environment: production
        WARNING: This is a development server. Do not use it in a production deployment.
        Use a production WSGI server instead.
        * Debug mode: on>

        to access the application click on the above http url which opens up in you default browser.
        contact Sharad or Shubham if you face any issues while running this.


    
    

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
