def reorder_quantiles(df):
    quantile_cols = ['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']
    # Sorting quantile values row-wise
    df_sorted = df.copy()
    for index, row in df.iterrows():
        sorted_quantiles = np.sort(row[quantile_cols])
        df_sorted.loc[index, quantile_cols] = sorted_quantiles
    return df_sorted