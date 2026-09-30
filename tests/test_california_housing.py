import unittest
import numpy as np
import pandas as pd
from portfolio.california_housing import FEATURES, validate_frame


class CaliforniaDataTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(np.ones((20640, 9)), columns=FEATURES + ['MedHouseVal'])

    def test_schema_and_row_count_are_required(self):
        frame = self.fixture()
        self.assertIs(validate_frame(frame), frame)
        for bad in (frame.iloc[:-1], frame.rename(columns={'MedHouseVal': 'SalePrice'})):
            with self.assertRaises(ValueError):
                validate_frame(bad)

    def test_nonfinite_features_and_nonpositive_target_rejected(self):
        for column, value in [('Latitude', np.nan), ('AveRooms', np.inf), ('MedHouseVal', 0)]:
            frame = self.fixture()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_frame(frame)
