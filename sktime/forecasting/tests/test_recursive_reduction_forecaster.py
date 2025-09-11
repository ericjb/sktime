from forecasting.compose._reduce import RecursiveReductionForecaster
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import product

def create_mock_data(n_samples=10, hierarchy=True):
    """Generates mock data for testing, with or without a hierarchical index"""
    dates = pd.period_range("2021-01", periods=n_samples, freq="M")

    if hierarchy:
        # Creating a hierarchical dataset with two levels (X, Y)
        data = [(x, y, date, np.random.uniform(10, 20000)) 
                for x, y, date in product(["X1", "X2"], ["Y1", "Y2"], dates)]
        df = pd.DataFrame.from_records(data, columns=["X", "Y", "date", "value"])

        # Aggregate across the second level (Y) to create a "__total" entry
        df_total = df.groupby(["X", "date"], as_index=False)["value"].sum().assign(Y="__total")
        df = pd.concat([df, df_total])

        # Set MultiIndex
        df = df.set_index(["X", "Y", "date"]).sort_index()

    else:
        # **Edge case: Simple time series without hierarchy**
        data = [(date, np.random.uniform(10, 20000)) for date in dates]
        df = pd.DataFrame.from_records(data, columns=["date", "value"])

        # No MultiIndex needed
        df = df.set_index("date").sort_index()

    return df

def test_recursive_reduction_forecaster_init():
    """Tests whether the forecaster initializes with the correct parameters"""
    forecaster = RecursiveReductionForecaster(
        estimator=LinearRegression(), window_length=5, impute_method="bfill", pooling="local"
    )
    
    assert forecaster.window_length == 5
    assert isinstance(forecaster.estimator, LinearRegression)
    assert forecaster.impute_method == "bfill"
    assert forecaster.pooling == "local"

def test_recursive_reduction_forecaster_fit():
    """Checks if the forecaster correctly fits the model"""
    df = create_mock_data()
    forecaster = RecursiveReductionForecaster(estimator=LinearRegression(), window_length=5)
    
    forecaster.fit(df)

    # Ensure the model is marked as fitted
    assert hasattr(forecaster, "_is_fitted"), "Forecaster is missing the '_is_fitted' attribute"
    assert forecaster._is_fitted is True, "Forecaster did not fit correctly"

def test_recursive_reduction_forecaster_predict_in_sample():
    """Tests if the forecaster correctly predicts within the training period"""
    df = create_mock_data()
    forecaster = RecursiveReductionForecaster(estimator=LinearRegression(), window_length=5)
    
    forecaster.fit(df)
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    unique_combinations = df.index.droplevel("date").nunique()
    
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * unique_combinations
    assert y_pred.shape[1] == 1  # Expecting a single column of predictions

def test_recursive_reduction_forecaster_predict_out_of_sample():
    """Checks if the forecaster correctly predicts beyond the training period"""
    df = create_mock_data()
    forecaster = RecursiveReductionForecaster(estimator=LinearRegression(), window_length=5)
    
    forecaster.fit(df)
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    unique_combinations = df.index.droplevel("date").nunique()
    
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape[0] == len(fh) * unique_combinations
    assert y_pred.shape[1] == 1

def test_recursive_reduction_forecaster_pooling():
    """Verifies that different pooling strategies work as expected"""
    df = create_mock_data()
    estimator = LinearRegression()
    fh = [1, 2, 3]
    
    for pooling in ["local", "global"]:
        forecaster = RecursiveReductionForecaster(estimator=estimator, window_length=5, pooling=pooling)
        forecaster.fit(df)
        y_pred = forecaster.predict(fh=fh)
        
        unique_combinations = df.index.droplevel("date").nunique()
        assert isinstance(y_pred, pd.DataFrame)
        assert y_pred.shape[0] == len(fh) * unique_combinations

def test_recursive_reduction_forecaster_hierarchy():
    """Ensures the forecaster correctly handles hierarchical structures"""
    df = create_mock_data(n_samples=20)
    forecaster = RecursiveReductionForecaster(estimator=LinearRegression(), window_length=5)

    forecaster.fit(df)
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    unique_combinations = df.index.droplevel("date").nunique()
    
    assert y_pred.shape[0] == len(fh) * unique_combinations, "Not all hierarchy nodes were predicted"

    unique_predictions = y_pred.reset_index().groupby(["X", "Y"])["value"].nunique()
    assert (unique_predictions > 1).any(), "All hierarchy nodes have identical predictions, which is unexpected"

    # Ensure each node has its own model
    if hasattr(forecaster, "models_"):
        assert len(forecaster.models_) == unique_combinations, "Each node should have a separate model"

def test_recursive_reduction_forecaster_global_one_level():
    """Checks if 'global' pooling is correctly handled when there's no hierarchy"""
    df = create_mock_data(n_samples=15, hierarchy=False)
    fh = [1, 2, 3]
    
    forecaster = RecursiveReductionForecaster(estimator=LinearRegression(), window_length=5, pooling="global")
    forecaster.fit(df)
    y_pred = forecaster.predict(fh=fh)

    # Ensure pooling is switched to 'local' when hierarchy is absent
    assert forecaster.pooling == "local", "Pooling should be automatically set to 'local'"
    assert isinstance(y_pred, (pd.Series, pd.DataFrame)), "Predictions should be a pandas object"

def test_rrf_nonnative_index():
    """Tests whether RRF can handle different index types."""
    df = create_mock_data()
    df.index = pd.period_range("2021-01", periods=len(df), freq="M")  # Test with PeriodIndex
    forecaster = RecursiveReductionForecaster(estimator=LinearRegression(), window_length=5)
    forecaster.fit(df)
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)
    assert isinstance(y_pred.index, pd.PeriodIndex)
