import pandas as pd

from functions.prepare_data import split_time
from functions.evaluation import evaluate_horizon
from tqdm import tqdm

def evaluate_daxmodel(model1, model2, df, last_x=100, years=False, months=False, weeks=False):
    '''
    model1, model2
        forecasting models (dict containing of name and function for 5 forecasts)
    df
        data frame containing energy data of last wednesday as last data point
    last_x
        number of iterations calculating score
    years, months, weeks
        set time intervals for iterations
    '''

    df_before = df
    evaluation_model1 = pd.DataFrame()
    evaluation_model2 = pd.DataFrame()

    for w in tqdm(range(2, last_x), desc="Evaluating models"):
        df_before, df_after = split_time(
            df_before, num_years=years, num_months=months, num_weeks=weeks)

        # Evaluate model1
        pred_model1 = model1['function'](df_before)
        evaluation_model1 = evaluate_and_append(evaluation_model1, pred_model1, df)

        # Evaluate model2
        pred_model2 = model2['function'](df_before)
        evaluation_model2 = evaluate_and_append(evaluation_model2, pred_model2, df)

    return evaluation_model1, evaluation_model2


def evaluate_and_append(evaluation_df, pred, df):
    obs = pd.DataFrame(columns=['ret1'])
    for index, row in pred.iterrows():
        print(row)
        if index in df.index:
            obs.loc[index] = df.loc[index]['ret1']

    merged_df = pd.merge(pred, obs, left_index=True, right_index=True)
    for index, row in merged_df.iterrows():
        quantile_preds = row[['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']]
        observation = row['ret1']
        score = evaluate_horizon(quantile_preds, observation)
        merged_df.at[index, 'score'] = score

    evaluation_df = pd.concat([evaluation_df, merged_df])
    return evaluation_df
