import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functions.prepare_data import split_time
from functions.evaluation import evaluate_horizon



def evaluate(model1, model2, df, start_date, end_date, horizon_format='days', last_x=5, years=False, months=False, weeks=True):
    '''
    Evaluate models over a specified range of dates for either DAX or energy forecasts.
    '''
    # Ensure df.index is timezone-aware
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')
    else:
        df.index = df.index.tz_convert('Europe/Berlin')

    date_range = pd.date_range(start=start_date, end=end_date, tz='Europe/Berlin')
    evaluation_model1 = pd.DataFrame()
    evaluation_model2 = pd.DataFrame()

    for current_date in date_range:
        # Data for model input: up to current_date
        df_model_input = df[df.index <= current_date]

        # Evaluate model1
        pred_model1 = model1['function'](df_model_input, date_str=current_date.strftime('%Y-%m-%d'))
        pred_model1 = prepare_predictions(pred_model1, df, horizon_format)
        evaluation_model1 = pd.concat([evaluation_model1, pred_model1])

        # Evaluate model2
        pred_model2 = model2['function'](df_model_input, date_str=current_date.strftime('%Y-%m-%d'))
        pred_model2 = prepare_predictions(pred_model2, df, horizon_format)
        evaluation_model2 = pd.concat([evaluation_model2, pred_model2])

    # Process evaluation results
    evaluation_model1['model'] = model1['name']
    evaluation_model2['model'] = model2['name']
    combined_evaluation = process_evaluation_results(evaluation_model1, evaluation_model2)

    return evaluation_model1, evaluation_model2, combined_evaluation


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


def process_evaluation_results(evaluation_model1, evaluation_model2):
    '''
    Process and plot the combined evaluation results.
    '''
    combined_evaluation = pd.concat([evaluation_model1, evaluation_model2])
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
