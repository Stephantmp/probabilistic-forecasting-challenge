
import pandas as pd

from functions.prepare_data import split_time
from functions.evaluation import evaluate_horizon
def evaluate(model1, model2, df, start_date, end_date, last_x=5, years=False, months=False, weeks=True):
    '''
    Evaluate DAX models over a specified range of dates.
    Parameters:
    - model1, model2: Models to evaluate.
    - df: DataFrame containing the DAX data.
    - start_date, end_date: Start and end dates of the evaluation period.
    - last_x, years, months, weeks: Parameters to control the time split.
    '''
    date_range = pd.date_range(start=start_date, end=end_date)
    evaluation_model1 = pd.DataFrame()
    evaluation_model2 = pd.DataFrame()

    for current_date in date_range:
        df_before = df.copy()
        date_str = current_date.strftime('%Y-%m-%d')

        # Evaluate model1
        pred_model1 = model1['function'](df_before, date_str=date_str)
        evaluation_model1 = evaluate_and_append(evaluation_model1, pred_model1, df)

        # Evaluate model2
        pred_model2 = model2['function'](df_before, date_str=date_str)
        evaluation_model2 = evaluate_and_append(evaluation_model2, pred_model2, df)

        # Update df_before for the next iteration
        df_before, _ = split_time(df_before, num_years=years, num_months=months, num_weeks=weeks)
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot data

    # Concatenate the two models' evaluations if needed
    evaluation_model1['model'] = 'Baseline Model'
    evaluation_model2['model'] = 'Quantile Regression'
    combined_evaluation = pd.concat([evaluation_model1, evaluation_model2])

    # Convert 'actual_forecast_date' to datetime if it's not already
    combined_evaluation['actual_forecast_date'] = pd.to_datetime(combined_evaluation['actual_forecast_date'])
    combined_evaluation = combined_evaluation.sort_values(by='actual_forecast_date')
    combined_evaluation.dropna(subset=['score'], inplace=True)
    combined_evaluation['actual_forecast_date'] = pd.to_datetime(combined_evaluation['actual_forecast_date'])
    combined_evaluation['score'] = combined_evaluation['score'].astype(float)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_evaluation, x='actual_forecast_date', y='score', hue='model')
    plt.title('Model Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Score')
    plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better readability
    plt.show()
    return evaluation_model1, evaluation_model2


# Example usage
start_date = '2023-01-01'
end_date = '2023-01-31'


def evaluate_and_append(evaluation_df, pred, df):
    # Convert forecast_date to datetime if it's not already
    pred['forecast_date'] = pd.to_datetime(pred['forecast_date'])

    # Function to calculate the actual date of forecast
    def calculate_actual_forecast_date(row):
        forecast_date = row['forecast_date']
        horizon = pd.Timedelta(row['horizon'])
        return forecast_date + horizon

    # Apply the function to calculate the actual forecast date
    pred['actual_forecast_date'] = pred.apply(calculate_actual_forecast_date, axis=1)

    # Convert the timezone of actual_forecast_date to match that of df
    pred['actual_forecast_date'] = pred['actual_forecast_date'].dt.tz_localize('Europe/Berlin')

    # Merge predictions with actual data based on the actual forecast date
    merged_df = pd.merge(pred, df, left_on='actual_forecast_date', right_index=True, how='left')

    # Debug: Check merged DataFrame
    print("Merged DataFrame in evaluate_and_append:\n", merged_df)


    for index, row in merged_df.iterrows():
        quantile_preds = row[['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']]
        observation = row['ret1']
        score = evaluate_horizon(quantile_preds, observation)
        merged_df.at[index, 'score'] = score

    evaluation_df = pd.concat([evaluation_df, merged_df])


    return evaluation_df