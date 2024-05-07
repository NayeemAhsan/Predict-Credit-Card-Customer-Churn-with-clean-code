# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The main objective iof this project is to predict credit car customers who are at a high risk of churning. 
This has been implemented adhering clean code best practices.
The project tries to accomplish the following steps:
- Load and clean the dataset
- Prepare data for training (feature engineering and preprocessing)
- Train for models. Since this is a classification problem, we used two classification models (random forest and logistic regression) for training 
- Save best models with their performance metrics
- Identify most important features influencing the predictions and visualize their impact using SHAP library

All scripts (churn_library.py, churn_library_test_logging.py, churn_library_test_pytest.py) were adjusted to the PEP8 standard using autopep8 module. In addition they all score above 8.0 using pylint clean code module.


## Files and data description
The project has the following directory architecture:
- Folders
    - Data --> contains the dataset     
    - images       
        - eda       --> contains EDA image output
        - results   --> contains image output of results like model scores, feature importance plots, etc. 
    - models        --> contains saved models in .pkl format
    - logs          --> log generated druing testing of library.py file

- project files 
    - churn_library.py 
    - churn_notebook.ipnyb
    - churn_library_test_pytest.py --> test script to run churn_library.py function using pytest that will show live logging information
    - churn_library_test_logging.py --> test script to run churn_library.py function that will store logging information
    - conftest.py --> pytest configuration file which contains common fixtures and variables - stored as Namespaces, to be used when called by the churn_library_test_pytest.py script.  
    - requirements_py3.8.txt --> required packages to run the script
    - Guide.ipynb --> project guideline for the nanodegree program


## Running Files
- The project should be executed with python 3.8 and the appropriate python packages
- The required packages are provided in the requirements_py3.8.txt file
- To run the project, execute the script `python churn_library.py` from the project folder
- Alternatively, the project can be executed using the jupyter notebook (churn_notebook.ipnyb) for a step-by-step approach
- The project script churn_library.py can be tested using either of the two test scripts:
    - churn_library_test_pytest.py can be run with this commad: pytest churn_library_test_pytest.py
    - churn_library_test_logging.py can be run with this command: python churn_library_test_logging.py  




