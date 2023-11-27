import pandas as pd

def ensemble(df_model1, df_model2, weight1, weight2):
    """
    Combine forecasts from two models into an ensemble forecast using specified weights.
    This version of the function works with DataFrames that may have additional columns.

    :param df_model1: DataFrame with forecasts from the first model
    :param df_model2: DataFrame with forecasts from the second model
    :param weight1: Weight for the first model's forecast
    :param weight2: Weight for the second model's forecast
    :return: DataFrame with the ensemble forecast
    """
    if df_model1.shape != df_model2.shape:
        raise ValueError("The shape of both DataFrames must be the same.")

    # Identify quantile columns by their names
    quantile_columns = [col for col in df_model1.columns if col.startswith('q')]

    # Ensuring the forecast dates, targets, and horizons match
    matching_columns = ['forecast_date', 'target', 'horizon']
    if not (df_model1[matching_columns] == df_model2[matching_columns]).all().all():
        raise ValueError("The forecast dates, targets, and horizons must match in both DataFrames.")

    # Calculating the ensemble forecast for quantile columns
    ensemble = df_model1[quantile_columns] * weight1 + df_model2[quantile_columns] * weight2

    # Adding back other columns from the first DataFrame
    for col in df_model1.columns:
        if col not in quantile_columns:
            ensemble[col] = df_model1[col]

    return ensemble

# Example usage
# df_model1 and df_model2 are your DataFrames containing the forecasts from the two models
# ensemble_df = ensemble(df_model1, df_model2, 0.5, 0.5)



