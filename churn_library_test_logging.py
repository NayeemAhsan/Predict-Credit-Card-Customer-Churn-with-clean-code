# test script doc string
"""
Test script to test Predict Credit Card Customer Chrun python file.
Running this script will create log file to check issues.

This script does not utilize pytest libraries. To see the pytest live logging, run the alternate test script (churn_library_test_pytest.py).

Use the following command to run the script:
python churn_library_test_logging.py

Author: Nayeem Ahsan
Date: 5/6/2024
"""
import os
import logging
import churn_library as cls

# Create the logs folder if it doesn't exist
logs_folder = './logs'
os.makedirs(logs_folder, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(logs_folder, 'churn_library.log'),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(file_pth):
    '''
    test data import
    Parameters:
        file_pth: Path to the CSV file (str)
    '''
    try:
        dframe = cls.import_data(file_pth)
        logging.info("Testing import_data: SUCCESS")

        try:
            assert dframe.shape[0] > 0
            assert dframe.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")


def test_data_cleaning(file_pth):
    '''
    test data cleaning function
    Parameters:
        dataframe: dataset 
    '''
    # import data
    df = cls.import_data(file_pth)
    
    try:
        # Call the data_cleaning function
        df_cleaned = cls.data_cleaning(df)
   
        try:
            # Check if the returned DataFrame has the correct number of columns
            assert df_cleaned.shape[1] == df.shape[1] - 1  # Two columns removed and one column added
            logging.info("Two columns were removed and one column was added")
        except AssertionError as err:
            logging.error(
                "Testing data_cleaning: The updated dataframe doesn't seem to have the updated number of columns. Check if the data_cleaning fucntion was able to remove two existing columns and add a new column.")
            raise err        
        try:
            # Check if the added column is present in the cleaned DataFrame
            assert 'Churn' in df_cleaned.columns
            logging.info("The column 'Churn' was successfully added")
        except AssertionError as err:
            logging.error(
                "Testing data_cleaning: The 'Churn' column was not added as a new column.")
            raise err        
        try:
            # Check if the deleted columns are not present in the cleaned DataFrame
            assert 'Attrition_Flag' not in df_cleaned.columns
            assert 'CLIENTNUM' not in df_cleaned.columns
            logging.info("Two columns were successfully removed")
        except AssertionError as err:
            logging.error(
                "Testing data_cleaning: The removal of the two columns were not successful")
            raise err        
        try:
            # Check if the values in the 'Churn' column are either 0 or 1
            assert all(df_cleaned['Churn'].isin([0, 1]))
            # Check if the 'Churn' column is correctly calculated based on 'Attrition_Flag'
            assert all(df_cleaned['Churn'] == (df_cleaned['Attrition_Flag'] == 'Attrited Customer').astype(int))
            logging.info("the 'Churn' column are either 0 or 1 and it is correctly calculated based on 'Attrition_Flag'")
        except AssertionError as err:
            logging.error(
                "Testing data_cleaning: The 'Churn' columns does't seem to have the correct values. They should have either 0 or 1, and should be correctly calculated based on 'Attrition_Flag'")
            raise err
        
        logging.info("Testing data_cleaning: SUCCESS")
    except Exception as err:
        logging.error("Testing data_cleaning function was not successful")


def test_eda(file_pth):
    '''
    test perform eda function
    Parameters:
        file_pth: Path to the CSV file (str)
    '''
    #import and clean the dataset
    dframe = cls.data_cleaning(cls.import_data(file_pth))

    try:
        cls.perform_eda(dframe)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: Failed - Error type %s", type(err))
        raise err


def test_encoder_helper(file_pth, category_lst):
    '''
    test encoder helper
    Parameters:
        file_pth: Path to the CSV file (str)
        category_lst (list): List of columns that contain categorical features.
    '''
    #import and clean the dataset
    dframe = cls.data_cleaning(cls.import_data(file_pth))

    try:
        response = 'Churn'
        newdframe = cls.encoder_helper(dframe, category_lst, response)

        logging.info("Testing encoder_helper is Successful with %s", category_lst)
    
        # Check if all categorical columns were encoded
        categorical_columns = newdframe.select_dtypes(include='object').columns.tolist()
        assert not categorical_columns, "At least one categorical column was NOT encoded - Check categorical features submitted"
        logging.info("All categorical columns were encoded")


    except KeyError as err:
        logging.error("Testing encoder_helper with %s failed: Check for categorical features not in the dataset", category_lst)
        raise err

    except Exception as err:
        logging.error("Testing encoder_helper failed - Error type %s", type(err))
        raise err


def test_perform_feature_engineering(file_pth):
    '''
    test perform_feature_engineering
    Parameters:
        file_pth: Path to the CSV file (str)
    '''
    #import and clean the dataset
    dframe = cls.data_cleaning(cls.import_data(file_pth))

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


def test_train_models(file_pth):
    '''
    test train_models
    Parameters:
        file_pth: Path to the CSV file (str)
    '''
    #import and clean the dataset
    dframe = cls.data_cleaning(cls.import_data(file_pth))

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

    #test data import
    test_import("./data/bank_data.csv")
    test_import("./data/no_file.csv")
    print('Finished testing data import')

    #test EDA
    #import matplotlib and set the backend to 'Agg'.
    import matplotlib
    matplotlib.use('Agg')
    test_eda("./data/bank_data.csv")
    print('Finished testing EDA')

    #test encoder helper function
    test_encoder_helper("./data/bank_data.csv", ['Gender','Education_Level','Income_Category','Card_Category', 'No_column_name'])
    test_encoder_helper("./data/bank_data.csv", ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'])
    print('Finished testing encoder_helper function')

    #test feature engineering function
    test_perform_feature_engineering("./data/bank_data.csv")
    print('Finished testing feature engineering function')

    #test train models
    #test_train_models("./data/bank_data.csv")
    #print('Finished testing train_models function')
