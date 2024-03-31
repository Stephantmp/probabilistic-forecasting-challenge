import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functions.evaluation import evaluate_horizon

def evaluate(*models, df, start_date, end_date, horizon_format='days'):
    '''
    Evaluate an arbitrary number of models over a specified range of dates for either DAX or energy forecasts,
    specifically focusing on forecasts with Wednesdays as the forecasting date.
    '''
    # Adjust here to check if the index is already timezone-aware
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')
    else:
        df.index = pd.to_datetime(df.index).tz_convert('Europe/Berlin')

    date_range = pd.date_range(start=start_date, end=end_date, tz='Europe/Berlin')
    wednesdays = date_range[date_range.weekday == 3]

    evaluations = []


    for model in models:
        evaluation_model = pd.DataFrame()
        for current_date in wednesdays:
            df_model_input = df[df.index <= current_date]
            pred = model['function'](df_model_input, date_str=current_date.strftime('%Y-%m-%d'))
            pred = prepare_predictions(pred, df, horizon_format)
            evaluation_model = pd.concat([evaluation_model, pred])

        evaluation_model['model'] = model['name']
        evaluations.append(evaluation_model)

    combined_evaluation = process_evaluation_results(*evaluations)
    return evaluations, combined_evaluation


def prepare_predictions(pred, df, horizon_format):
    '''
    Prepare and merge predictions with the actual data.
    '''
    pred['forecast_date'] = pd.to_datetime(pred['forecast_date'])
    pred['actual_forecast_date'] = pred.apply(lambda row: calculate_actual_forecast_date(row, horizon_format), axis=1)

    if not pred['actual_forecast_date'].dt.tz:
        pred['actual_forecast_date'] = pred['actual_forecast_date'].dt.tz_localize('Europe/Berlin')

    df.index = pd.to_datetime(df.index)
    if df.index.tz:
        df.index = df.index.tz_convert('Europe/Berlin')
    else:
        df.index = df.index.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')

    merged_df = pd.merge(pred, df, left_on='actual_forecast_date', right_index=True, how='left')

    for index, row in merged_df.iterrows():
        quantile_preds = row[['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']]
        observation = row['gesamt']
        score = evaluate_horizon(quantile_preds, observation)
        merged_df.at[index, 'score'] = score

    return merged_df


def calculate_actual_forecast_date(row, horizon_format):
    '''
    Calculate the actual date of forecast based on the horizon.
    '''
    forecast_date = row['forecast_date']
    horizon = row['horizon']

    if horizon_format == 'days':
        additional_time = pd.Timedelta(days=int(horizon.split()[0]))
    elif horizon_format == 'hours':
        additional_time = pd.Timedelta(hours=int(horizon.split()[0]))
    else:
        raise ValueError("Invalid horizon format. Please choose 'days' or 'hours'.")
    return forecast_date + additional_time

def process_evaluation_results(*evaluations):
    '''
    Process and plot the combined evaluation results from multiple models.
    '''
    combined_evaluation = pd.concat(evaluations)
    combined_evaluation['actual_forecast_date'] = pd.to_datetime(combined_evaluation['actual_forecast_date'])
    combined_evaluation = combined_evaluation.sort_values(by='actual_forecast_date')
    combined_evaluation.dropna(subset=['score'], inplace=True)
    combined_evaluation['score'] = combined_evaluation['score'].astype(float)

    # Define a custom palette
    custom_palette = ['red', 'green', 'blue', 'purple', 'orange']

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_evaluation, x='actual_forecast_date', y='score', hue='model',palette=custom_palette)
    plt.title('Model Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Pinball Loss')
    plt.xticks(rotation=45)
    plt.show()

    grouped_scores = combined_evaluation.groupby(['model', 'horizon'])['score'].agg(['mean', 'median', 'std'])
    return grouped_scores
