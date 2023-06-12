import pandas as pd


def csv_to_excel(csv_file_name, excel_file_name):
    # read csv file
    df = pd.read_csv(csv_file_name)
    # convert to excel
    df.to_excel(excel_file_name, index=False)


csv_file_name = 'resource_log.csv'
excel_file_name = 'output.xlsx'

csv_to_excel(csv_file_name, excel_file_name)
