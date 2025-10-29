import pandas as pd 

# Validated that we used the same preprocessing step when we created the GPT preprocessing file

class CheXpertPreprocessing():
    '''Class perfroms preprocessing on a dataset given the path to the CSV file of the dataset'''
    def __init__(self, path, splits):
        self.path = path
        self.splits = splits

    def read_data(self):
        '''Read data from path and filter the dataset'''
        chexpert_plus = pd.read_csv(self.path)
        chexpert_plus = chexpert_plus.drop("split", axis=1)
        self.splits["path"] = self.splits["path"].str.replace("CheXpert-v1.0/", "", regex=False)
        chexpert_plus = chexpert_plus.merge(self.splits, left_on="path_to_image", right_on="path", how="inner")
        return chexpert_plus