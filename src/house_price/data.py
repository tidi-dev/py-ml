"""Load the California Housing dataset.

Each row describes one California *census block group* (a small
neighbourhood), not one individual house. See docs/01-understanding-the-data.md.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing

# The column we want to predict: median house value, in units of $100,000.
TARGET_COLUMN = "MedHouseVal"

# The eight columns that describe each block group, in dataset order.
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_housing_data() -> pd.DataFrame:
    """Return the California Housing dataset as a single DataFrame.

    The DataFrame has 20,640 rows (one per block group) and 9 columns:
    the 8 feature columns followed by the ``MedHouseVal`` target column.

    The first call downloads the data (about 400 KB) and caches it in
    ``~/scikit_learn_data``; later calls read the cached copy.
    """
    housing = fetch_california_housing(as_frame=True)
    return housing.frame
