import numpy as np


class ZScoreNormalizer:
    def __init__(self):
        self.features_shape = (0, )
        self.features_mask = np.ma.nomask
        self.mean = 0
        self.std = 0

    def __str__(self):
        return f"ZScoreNormalizer(features_shape: {self.features_shape} mean: {self.mean} standard deviation: {self.std})"
    
    def __repr__(self):
        return self.__str__()

    def fit(self, data, ignore_features = []):
        samples_n = data.shape[0]
        self.features_shape = data.shape[1:]

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

        if len(ignore_features) > 0:
            mask = np.zeros(self.features_shape, dtype=np.bool)
            mask[ignore_features] = True
            self.features_mask = mask
        else:
            self.features_mask = np.ma.nomask

        self.mean = np.ma.masked_array(self.mean, mask=self.features_mask)
        self.std = np.ma.masked_array(self.std, mask=self.features_mask)

    def transform(self, data, features=slice(None)):
        data_t = (data - self.mean[features])/self.std[features]
        return np.ma.filled(data_t, fill_value = data)

    def inverse_transform(self, data, features=slice(None)):
        data_invt = data * self.std[features] + self.mean[features]
        return np.ma.filled(data_invt, data)
    
