import pytest
import churn_library as cls

@pytest.fixture(scope="module")
def file_pth():
    """Fixture to provide a file path."""
    return "./data/bank_data.csv"

@pytest.fixture(scope="module")
def dataframe(file_pth):
    """Fixture to provide the DataFrame loaded from the file."""
    return cls.import_data(file_pth)

# Creating a Dataframe object in Namespace
@pytest.fixture(scope='module')
def df(dataframe):
    df = cls.data_cleaning(dataframe)
    return df
