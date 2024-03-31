import pandas as pd


def ensemble(df_models, weights):
    """
    Combine forecasts from multiple models into an ensemble forecast using specified weights.
    This version of the function works with DataFrames that may have additional columns.

    :param df_models: List of DataFrames with forecasts from each model
    :param weights: List of weights for each model's forecast
    :return: DataFrame with the ensemble forecast
    """
    if len(df_models) != len(weights):
        raise ValueError("The number of DataFrames and weights must be the same.")

    # Ensuring all DataFrames have the same shape
    first_shape = df_models[0].shape
    if not all(df.shape == first_shape for df in df_models):
        raise ValueError("The shape of all DataFrames must be the same.")

    # Identify quantile columns by their names
    quantile_columns = [col for col in df_models[0].columns if col.startswith('q')]

    # Ensuring the forecast dates, targets, and horizons match in all DataFrames
    matching_columns = ['forecast_date', 'target', 'horizon']
    for df in df_models[1:]:
        if not (df_models[0][matching_columns] == df[matching_columns]).all().all():
            raise ValueError("The forecast dates, targets, and horizons must match in all DataFrames.")

    # Calculating the ensemble forecast for quantile columns
    ensemble = sum([df[quantile_columns] * weight for df, weight in zip(df_models, weights)])

    # Adding back other columns from the first DataFrame
    for col in df_models[0].columns:
        if col not in quantile_columns:
            ensemble[col] = df_models[0][col]

    ensemble = ensemble[df_models[0].columns]
    return ensemble

# Example usage
# df_models = [df_model1, df_model2, df_model3]  # Your DataFrames containing the forecasts
# weights = [0.5, 0.3, 0.2]  # Weights for each model's forecast
# ensemble_df = ensemble(df_models, weights)



