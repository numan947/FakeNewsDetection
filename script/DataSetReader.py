import sys
sys.path.append('./')

"""
This file contains code for reading various datasets used in the project.
The datasets are read using pandas library.
The datasets are read from the paths defined in `DataSetPaths.py` file
and the data is returned as pandas dataframe.
"""


import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import DataSetPaths

"""
The function reads the liar dataset from the paths defined in `DataSetPaths.py` file.
"""
def read_liar_data():
    
    columns = [
                "id",
                "label",
                "statement",
                "subject",
                "speaker",
                "job_title",
                "state_info",
                "party_affiliation",
                "barely_true_counts",
                "false_counts",
                "half_true_counts",
                "mostly_true_counts",
                "pants_on_fire_counts",
                "context"
            ]

    train_data = pd.read_csv(DataSetPaths.LIAR_TRAIN_DATA, sep='\t',names=columns, header=None)
    valid_data = pd.read_csv(DataSetPaths.LIAR_VALID_DATA, sep='\t',names=columns, header=None)
    test_data = pd.read_csv(DataSetPaths.LIAR_TEST_DATA, sep='\t',names=columns, header=None)
    
    train_data = train_data[['label', 'statement']]
    valid_data = valid_data[['label','statement']]
    test_data = test_data[['label','statement']]
    
    return train_data, valid_data, test_data
"""
The following test function tests the `read_liar_data` function.
"""
def test_read_liar_data():
    train_data, valid_data, test_data = read_liar_data()
    
    print("STATS -- Data Sample:\n\n")
    print(train_data.head())
    print(valid_data.head())
    print(test_data.head())
    
    print("\n\nSTATS -- Shape of Data:\n\n")
    print("Train Data: ", train_data.shape)
    print("Valid Data: ", valid_data.shape)
    print("Test Data: ", test_data.shape)
    
    
    print("\n\nSTATS -- Label Distribution:\n\n")
    print("Train Data: ", train_data['label'].value_counts())
    print("Valid Data: ", valid_data['label'].value_counts())
    print("Test Data: ", test_data['label'].value_counts())
    
    print("\n\nSTATS -- Label Distribution (Percentage):\n\n")
    print("Train Data: ", train_data['label'].value_counts(normalize=True))
    print("Valid Data: ", valid_data['label'].value_counts(normalize=True))
    print("Test Data: ", test_data['label'].value_counts(normalize=True))






## testing here
if __name__ == "__main__":
    test_read_liar_data()
    pass
    