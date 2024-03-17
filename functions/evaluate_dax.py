
import pandas as pd
import pytz

from functions.prepare_data import split_time
from functions.evaluation import evaluate_horizon


def evaluate(model1, model2, df, start_date, end_date, last_x=5, years=False, months=False, weeks=True):
    '''
    Evaluate DAX models over a specified range of dates.
    '''
    date_range = pd.date_range(start=start_date, end=end_date)
    # Filter date_range to include only Wednesdays
    wednesdays = [d for d in date_range if d.weekday() == 2]

    evaluation_model1 = pd.DataFrame()
    evaluation_model2 = pd.DataFrame()

    for current_date in wednesdays:
        # Adjust current_date timezone if necessary
        if df.index.tz is not None:
            current_date = current_date.tz_localize(df.index.tz)

        # Data for model input: up to current_date
        df_model_input = df[df.index < current_date]

        # Evaluate model1
        pred_model1 = model1['function'](df_model_input, date_str=current_date.strftime('%Y-%m-%d'))
        evaluation_model1 = evaluate_and_append(evaluation_model1, pred_model1, df)

        # Evaluate model2
        pred_model2 = model2['function'](df_model_input, date_str=current_date.strftime('%Y-%m-%d'))
        evaluation_model2 = evaluate_and_append(evaluation_model2, pred_model2, df)



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
    # Aggregating scores by model and horizon
    grouped_scores = combined_evaluation.groupby(['model', 'horizon'])['score'].agg(['mean', 'median', 'std'])
    grouped_scores
    return evaluation_model1, evaluation_model2


from datetime import timedelta
def add_business_days(from_date, add_days):
    current_date = from_date
    while add_days > 0:
        current_date += timedelta(days=1)
        # weekday() returns 0 for Monday, 6 for Sunday
        if current_date.weekday() < 5:  # Monday to Friday are considered
            add_days -= 1
    return current_date

def evaluate_and_append(evaluation_df, pred, df):
    # Convert forecast_date to datetime if it's not already
    pred['forecast_date'] = pd.to_datetime(pred['forecast_date'])

    # Function to calculate the actual date of forecast considering business days
    def calculate_actual_forecast_date(row):
        forecast_date = row['forecast_date']
        horizon_days = int(row['horizon'].split()[0])  # Convert the horizon to integer days
        actual_forecast_date = add_business_days(forecast_date, horizon_days)
        return actual_forecast_date

    # Apply the function to calculate the actual forecast date
    pred['actual_forecast_date'] = pred.apply(calculate_actual_forecast_date, axis=1)

    # Assuming your 'df' DataFrame's index is already localized to 'Europe/Berlin'
    # If not, you may need to localize or convert it accordingly
    pred['actual_forecast_date'] = pred['actual_forecast_date'].dt.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')

    # Merge predictions with actual data based on the actual forecast date
    merged_df = pd.merge(pred, df, left_on='actual_forecast_date', right_index=True, how='left')

    # Loop to evaluate predictions against actual observed values
    for index, row in merged_df.iterrows():
        horizon_days = int(row['horizon'].split()[0])
        ret_str = f'future_ret{horizon_days}'
        observation = row[ret_str]
        quantile_preds = row[['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']]
        score = evaluate_horizon(quantile_preds.values, observation)
        merged_df.at[index, 'score'] = score

    evaluation_df = pd.concat([evaluation_df, merged_df])
    return evaluation_df
