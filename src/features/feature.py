class FeatureEngineering:
    '''
    functions for:
        1. creating new features based on existing ones.
        2. Selecting a features subset from data. 
    '''
    def __init__(self, data):
        self.data = data

    def make_features():
        '''single function to create all types of features
        '''
        raise NotImplementedError

    def select_features():
        '''Perform CV to select features based on specified 
        techniques'''
        raise NotImplementedError