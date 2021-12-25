import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(path = 'data.csv'):
    df = pd.read_csv('data.csv')
    pd.set_option('display.max_columns', None)

    # Do not include unique indicator columns, such as the name of the fighters.
    # Unique indicator columns don't give any useful information to the model.
    df.drop(columns=['B_ID', 'B_Name', 'R_ID', 'R_Name', 'Event_ID', 'Fight_ID'], inplace=True)

    # df.info(max_cols=889)

    # Handle missing data for continuous columns only.
    # Assume we will use a model that can work with missing data.
    df.describe(include="object")

    # print(sum(df['Date'].isna()))

    df['Date'] = pd.to_datetime(df['Date'])

    # print(
    #     sum(df['B_HomeTown'].isna()),
    #     sum(df['B_Location'].isna()),
    #     sum(df['R_HomeTown'].isna()),
    #     sum(df['R_Location'].isna())
    # )

    # print(
    #     sum(df['B_HomeTown'] == df['B_Location']),
    #     sum(df['R_HomeTown'] == df['R_Location'])
    # )

    # print(df['B_Location'].value_counts(dropna=False))

    # print(df['B_HomeTown'].value_counts(dropna=False))

    df['B_moved'] = df['B_HomeTown'] != df['B_Location']

    df['B_HomeTown_country'] = df['B_HomeTown'].str.split(' ').str[-1]
    df['B_HomeTown_country'].replace('', np.nan, inplace=True)

    df['B_Location_country'] = df['B_Location'].str.split(' ').str[-1]
    df['B_Location_country'].replace('', np.nan, inplace=True)

    df['B_moved_country'] = df['B_HomeTown_country'] != df['B_Location_country']

    df['R_moved'] = df['R_HomeTown'] != df['R_Location']

    df['R_HomeTown_country'] = df['R_HomeTown'].str.split(' ').str[-1]
    df['R_HomeTown_country'].replace('', np.nan, inplace=True)

    df['R_Location_country'] = df['R_Location'].str.split(' ').str[-1]
    df['R_Location_country'].replace('', np.nan, inplace=True)

    df['R_moved_country'] = df['R_HomeTown_country'] != df['R_Location_country']


    def keep_common_categories(df, columns, percent=0.7, value='other'):
        for column in columns:
            l = len(df[column].unique())
            not_common_categories = df[column].value_counts(dropna=False)[int(l * percent):].index
            for c in not_common_categories:
                df[column].replace(c, value, inplace=True)


    columns = ['B_HomeTown', 'B_Location', 'B_HomeTown_country', 'B_Location_country',
               'R_HomeTown', 'R_Location', 'R_HomeTown_country', 'R_Location_country']
    keep_common_categories(df, columns)

    df['B_Location'].value_counts(dropna=False)

    # Bin the data for columns ‘R_Weight’ and ‘B_Weight’.
    # print(sum(df['R_Weight'].isna()), sum(df['B_Weight'].isna()))

    columns = ['R_Weight', 'B_Weight']
    bin_count = [3, 3]
    labels = [['lightweight', 'middleweight', 'heavyweight']] * 2
    for i, column in enumerate(columns):
        bins = np.linspace(df[column].min(), df[column].max(), bin_count[i] + 1)
        df[column + '_bins'] = pd.cut(df[column], bins=bins, labels=labels[i], include_lowest=True)

    # print(df[['B_Weight', 'B_Weight_bins', 'R_Weight', 'R_Weight_bins']])

    # Handling missing values

    df_int = df.select_dtypes(include=["integer"])
    # print(df_int.isna().sum())

    float_cols = df.select_dtypes(include=["floating"]).columns.values
    # print(df.select_dtypes(include=["floating"]).isna().sum().sort_values())

    threshold = int(df.shape[0] * 0.4)
    df.drop(columns=float_cols[df[float_cols].isna().sum() > threshold], inplace=True)


    # print(df.select_dtypes(include=["floating"]).isna().sum().sort_values())

    def fillna(df, columns, method='mean'):
        for column in columns:
            if method == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)


    columns = df.select_dtypes(include=["floating"]).isna().columns.values
    fillna(df, columns[:170])
    fillna(df, columns[170:], method='mode')

    # print(df.select_dtypes(include=["floating"]).isna().sum().sort_values())

    # print(df.isna().sum().sort_values()[::-1][:8])

    # Spliting and saving the data
    df = pd.get_dummies(data=df, drop_first=True)
    train_data, test_data = train_test_split(df)

    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)

if __name__ == '__main__':
    clean_data()