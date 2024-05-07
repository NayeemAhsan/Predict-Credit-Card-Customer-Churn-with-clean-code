# test script doc string
"""
Test script to test Predict Credit Card Customer Chrun python file.
This script utilizes pytest libraries. Running this will show pytest live logging. 

This script calls conftest.py file when running. conftest.py contains common fixtures ans store common variables in the Namespace
that will be used by most of the test modules.

Use the following command to run the script:
pytest churn_library_test_pytest.py

Author: Nayeem Ahsan
Date: 5/6/2024
"""
import os
import logging
import pytest
import churn_library as cls

# Configure logging
''' this script is created to test using pytest live logging. 
    So, running this script using python or pytest, will not create any log file. 
    To check log file, run the churn_library_test_logging.py script using python command'''
# Create the logs folder if it doesn't exist
logs_folder = './logs'
os.makedirs(logs_folder, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(logs_folder, 'churn_library.log'),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module", params=["./data/bank_data.csv", "./data/hospital_data.csv"])
def path_params(request):
    '''
    Fixture: path parameters will be used as arguments to test test_import function
    '''
    return request.param

def test_import(path_params):
    '''
    test data import
    Parameters:
        path_params: path parameters (file paths)
    '''    
    try:
        dframe = cls.import_data(path_params)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        
    try:
        dframe = cls.import_data(path_params)
        assert dframe.shape[0] > 0
        assert dframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
    

    
def test_data_cleaning(df, dataframe):
    '''
    test data cleaning function
    Parameters:
        dataframe: dataset 
    '''
    try:   
        dframe = df.copy()    
        try:
            # Check if the returned DataFrame has the correct number of columns
            assert dframe.shape[1] == dataframe.shape[1] - 1  # Two columns removed and one column added
            logging.info("Two columns were removed and one column was added")
        except AssertionError as err:
            logging.error(
                "Testing data_cleaning: The updated dataframe doesn't seem to have the updated number of columns. Check if the data_cleaning fucntion was able to remove two existing columns and add a new column.")
            raise err        
        try:
            # Check if the added column is present in the cleaned DataFrame
            assert 'Churn' in dframe.columns
            logging.info("The column 'Churn' was successfully added")
        except AssertionError as err:
            logging.error(
                "Testing data_cleaning: The 'Churn' column was not added as a new column.")
            raise err        
        try:
            # Check if the deleted columns are not present in the cleaned DataFrame
            assert 'Attrition_Flag' not in dframe.columns
            assert 'CLIENTNUM' not in dframe.columns
            logging.info("Two columns were successfully removed")
        except AssertionError as err:
            logging.error(
                "Testing data_cleaning: The removal of the two columns were not successful")
            raise err        
        try:
            # Check if the values in the 'Churn' column are either 0 or 1
            assert all(dframe['Churn'].isin([0, 1]))
            # Check if the 'Churn' column is correctly calculated based on 'Attrition_Flag'
            assert all(dframe['Churn'] == (dframe['Attrition_Flag'] == 'Attrited Customer').astype(int))
            logging.info("the 'Churn' column are either 0 or 1 and it is correctly calculated based on 'Attrition_Flag'")
        except AssertionError as err:
            logging.error(
                "Testing data_cleaning: The 'Churn' columns does't seem to have the correct values. They should have either 0 or 1, and should be correctly calculated based on 'Attrition_Flag'")
            raise err
        
        logging.info("Testing data_cleaning: SUCCESS")
    except Exception as err:
        logging.error("Testing data_cleaning function was not successful")


def test_eda(df):
    '''
    test perform eda function
    Parameters:
        file_pth: Path to the CSV file (str)
    '''
    # get dataframe from pytest Namespace
    dframe = df.copy()

    try:
        cls.perform_eda(dframe)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: Failed - Error type %s", type(err))
        raise err


@pytest.fixture(scope="module",
                params=[['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'],
                        ['Gender','Education_Level','Marital_Status','Income_Category'],
                        ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category','Not_a_column'],
                        ])
def encoder_params(request):
    """
    Fixture - The test_encoder_helper function will
    use the parameters returned by encoder_params() as arguments
    """
    return request.param


def test_encoder_helper(df, encoder_params):
    '''
    test encoder helper function
    Parameters:
        encoder_params: A list of categorical columns to be encoded. 
    '''
    # get dataframe from pytest Namespace
    dframe = df.copy()
    # get the encoder parameters
    cat_list = encoder_params
    response = "Churn"

    try:
        newdframe = cls.encoder_helper(dframe, cat_list, response)
        logging.info("Testing encoder_helper is Successful with %s", cat_list)
        
        # Check if all categorical columns were encoded
        categorical_columns = newdframe.select_dtypes(include='object').columns.tolist()
        assert not categorical_columns, "At least one categorical column was NOT encoded - Check categorical features submitted"
        logging.info("All categorical columns were encoded")

    except KeyError as err:
        logging.error("Testing encoder_helper with %s failed: Check for categorical features not in the dataset", cat_list)
        raise err

    except Exception as err:
        logging.error("Testing encoder_helper failed - Error type %s", type(err))
        raise err
   
  
def test_perform_feature_engineering(df):
    '''
    test perform_feature_engineering
    '''
    # get dataframe from pytest Namespace
    dframe = df.copy()
    try:
        train_x, test_x, train_y, test_y = cls.perform_feature_engineering(dframe)
        logging.info("Testing perform_feature_engineering - SUCCESS")

        assert train_x.shape[0] > 0
        assert train_x.shape[1] > 0
        assert test_x.shape[0] > 0
        assert test_x.shape[1] > 0
        assert train_y.shape[0] > 0
        assert test_y.shape[0] > 0
        logging.info("perform_feature_engineering returned Train / Test set of shape %s %s", train_x.shape, test_x.shape)

    except AssertionError:
            logging.error(
                "The returned train / test datasets do not appear to have rows and columns")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering failed - Error type %s", type(err))
        raise err


def test_train_models(df):
    '''
    test train_models
    '''
    # get dataframe from pytest Namespace
    dframe = df.copy()
    # perform feature engineering and split train and test data
    train_x, test_x, train_y, test_y = cls.perform_feature_engineering(
        dframe, 'Churn')

    try:
        cls.train_models(train_x, test_x, train_y, test_y)
        logging.info("Testing train_models: SUCCESS")
    except MemoryError as err:
        logging.error("Testing train_models: Out of memory while training models - Error type %s", type(err))
    except Exception as err:
        logging.error("Testing train_models failed - Error type %s", type(err))
        raise err


if __name__ == "__main__":
    pass