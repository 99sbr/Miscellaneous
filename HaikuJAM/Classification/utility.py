## Null Counting Fucntion
def null_values(df):
    import pandas as pd
    sum_null = df.isnull().sum()
    total = df.isnull().count()
    percent_null_values = 100 * sum_null / total
    df_null = pd.DataFrame()
    df_null['Total'] = total
    df_null['Null_Count'] = sum_null
    df_null['Percent'] = round(percent_null_values, 2)
    df_null = df_null.sort_values(by='Null_Count', ascending=False)
    df_null = df_null[df_null.Null_Count > 0]

    return df_null