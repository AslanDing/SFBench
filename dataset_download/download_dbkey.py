import os
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


# def generate_month_ranges(start_date, end_date):
#     start = pd.to_datetime(start_date, format="%d-%b-%Y")
#     end = pd.to_datetime(end_date, format="%d-%b-%Y")
#     return pd.date_range(start=start, end=end, freq='MS').strftime("%Y-%m").tolist()

def generate_month_ranges(start_date, end_date):
    start = pd.to_datetime(start_date, format="%d-%b-%Y")
    end = pd.to_datetime(end_date, format="%d-%b-%Y")
    months = pd.date_range(start=start, end=end, freq='MS')  # Start of each month
    return [date.strftime("%Y%m01") for date in months]  # First day of each month



def download_file(key,start_date,end_date,save_name):
    base_url = 'http://my.sfwmd.gov/dbhydroplsql/web_io.report_process?'

    format = 'v_report_type=format6'
    datetime = f'v_period=uspec&v_start_date={start_date}&v_end_date={end_date}'
    target = 'v_target_code=file_csv'
    run_mode = 'v_run_mode=onLine'
    js_flag = 'v_js_flag=Y'
    dbkey = 'v_dbkey='

    url = base_url + '&' + datetime + '&' + format + '&' + target + '&' + run_mode + '&' + js_flag + '&' + dbkey + key
    filename = save_name #f"{name}.csv"  # Specify the name to save the file as

    # Send a GET request to the URL and stream the content
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        # print("Download complete!")
    else:
        print("Failed to download the file.")


def main():
    download_dir = './'

    csv_file = './dbkey_tables.csv'
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        dbkey = row['Dbkey']
        start_date = row['Start Date']
        end_date = row['End Date']
        months = generate_month_ranges(start_date, end_date)

        dir =   download_dir + dbkey
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_name = dir + f'/{dbkey}.csv'
        for idx in range(1,len(months)):
            start_date = months[idx-1]
            end_date = months[idx]
            download_file(dbkey,start_date,end_date,save_name.replace(".csv",f"_{idx}.csv"))

        break

if __name__=="__main__":
    main()

