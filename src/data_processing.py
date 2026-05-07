import pandas as pd

def load_and_clean_data(path):
    df = pd.read_excel(r"C:\Users\Dipesh Paliwal\Desktop\microgcc assignment\data\Copy of Forecasting Case- Study.xlsx")


    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['State', 'Date'])

    return df