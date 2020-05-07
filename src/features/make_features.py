class MakeFeatures:
    '''
    Make features from existing ones.
    > wrapper around sklearn
    1. polynomial features : degree 2, 3
    2. trigonometry features : sin, cosine
    3. transformed features : Standard, MinMaxScaler, Log1p, Box-Cox
    4. interaction features : between 2 or more features
    5. Encoded features : OHE, LabelEncoded, target encoded, mean encoded, 
    6. un-skew features : Box-Cox, log
    7. lagged features : time lagged (based on acf, pacf, shift_by_n)
    8. Date features : all date features (from fastai)

    '''
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def polynomial_features():
        raise NotImplementedError

    def trigonometry_features():
        raise NotImplementedError

    def transformed_features():
        raise NotImplementedError

    def interaction_features():
        raise NotImplementedError

    def encoded_features():
        raise NotImplemented

    def unskew_features():
        raise NotImplemented

    def lagged_features():
        raise NotImplemented

    def date_features():
        raise NotImplemented