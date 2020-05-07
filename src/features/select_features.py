class SelectFeatures:
    '''
    Select features from data for modeling.
    > wrapper around sklearn 

    1. SFE
    2. p-value
    3. stepwise
    4. ppscore
    5. vif
    6. correlation
    7. 

    '''
    def __init__(self, data, target):
        self.data = data
        self.target = target