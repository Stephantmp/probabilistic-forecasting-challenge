import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functions.prepare_data import split_time
from functions.evaluation import evaluate_horizon

def evaluate(*models, df, start_date, end_date, horizon_format='days'):
    '''
    Evaluate an arbitrary number of models over a specified range of dates for either DAX or energy forecasts.
    '''
    # Ensure df.index is timezone-aware
    date_range = pd.date_range(start=start_date, end=end_date)
    wednesdays = [d for d in date_range if d.weekday() == 2]

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')
    else:
        df.index = df.index.tz_convert('Europe/Berlin')

    date_range = pd.date_range(start=start_date, end=end_date, tz='Europe/Berlin')
    evaluations = []

    for current_date in wednesdays:
        # Adjust current_date timezone if necessary
        if df.index.tz is not None:
            current_date = current_date.tz_localize(df.index.tz)
        # Data for model input: up to current_date
        df_model_input = df[df.index <= current_date]

        for model in models:
            # Evaluate model
            pred = model['function'](df_model_input, date_str=current_date.strftime('%Y-%m-%d'))
            pred = prepare_predictions(pred, df, horizon_format)
            pred['model'] = model['name']
            evaluations.append(pred)

    combined_evaluation = pd.concat(evaluations)
    grouped_scores = process_evaluation_results(combined_evaluation)

    return evaluations, grouped_scores

def prepare_predictions(pred, df, horizon_format):
    '''
    Prepare and merge predictions with the actual data.
    '''
    pred['forecast_date'] = pd.to_datetime(pred['forecast_date'])
    pred['actual_forecast_date'] = pred.apply(lambda row: calculate_actual_forecast_date(row, horizon_format), axis=1)

    # Localize 'actual_forecast_date' to 'Europe/Berlin' if it's not already localized
    if not pred['actual_forecast_date'].dt.tz:
        pred['actual_forecast_date'] = pred['actual_forecast_date'].dt.tz_localize('Europe/Berlin')

    # Localize or convert the dataframe index to 'Europe/Berlin'
    df.index = pd.to_datetime(df.index)
    if df.index.tz:
        df.index = df.index.tz_convert('Europe/Berlin')
    else:
        df.index = df.index.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')

    # Merge predictions with actual data based on the actual forecast date
    merged_df = pd.merge(pred, df, left_on='actual_forecast_date', right_index=True, how='left')

    # Calculate scores
    for index, row in merged_df.iterrows():
        quantile_preds = row[['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']]
        observation = row['gesamt']
        score = evaluate_horizon(quantile_preds, observation)
        merged_df.at[index, 'score'] = score

    return merged_df

def calculate_actual_forecast_date(row, horizon_format):
    '''
    Calculate the actual date of forecast.
    '''
    forecast_date = row['forecast_date']
    horizon = row['horizon']
    if horizon_format == 'days':
        horizon = pd.Timedelta(days=int(horizon.split()[0]))
    elif horizon_format == 'hours':
        horizon = pd.Timedelta(hours=int(horizon.split()[0]))
    else:
        raise ValueError("Invalid horizon format. Please choose 'days' or 'hours'.")
    return forecast_date + horizon

def process_evaluation_results(combined_evaluation):
    '''
    Process and plot the combined evaluation results for an arbitrary number of models.
    '''
    combined_evaluation['actual_forecast_date'] = pd.to_datetime(combined_evaluation['actual_forecast_date'])
    combined_evaluation = combined_evaluation.sort_values(by='actual_forecast_date')
    combined_evaluation.dropna(subset=['score'], inplace=True)
    combined_evaluation['score'] = combined_evaluation['score'].astype(float)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_evaluation, x='actual_forecast_date', y='score', hue='model')
    plt.title('Model Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.show()

    # Aggregating scores by model and horizon
    grouped_scores = combined_evaluation.groupby(['model', 'horizon'])['score'].agg(['mean', 'median', 'std'])

    return grouped_scores
