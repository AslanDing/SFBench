import requests

# the website :  https://apps.sfwmd.gov/WAB/EnvironmentalMonitoring/index.html
# docments : https://www.sfwmd.gov/sites/default/files/documents/dbhydrobrowseruserdocumentation.pdf

base_url = 'http://my.sfwmd.gov/dbhydroplsql/web_io.report_process?'

# db keys
db_keys = ['PT140']
# station name
name = ['MRMS4']
# water level

date_lists = [('20100101','20100131'),
              ('20100201','20100228'),
              ('20100301','20100331'),
              ('20100401','20100430'),
              ('20100501','20100531'),
              ('20100601','20100630'),
              ('20100701','20100731'),
              ('20100801','20100831'),
              ('20100901','20100930'),
              ('20101001','20101031'),
              ('20101101','20101130'),
              ('20101201','20101231')]

format = 'v_report_type=format6'
target = 'v_target_code=file_csv'
run_mode = 'v_run_mode=onLine'
js_flag = 'v_js_flag=Y'

datetime = 'v_period=uspec&v_start_date=%s&v_end_date=%s'
dbkey = 'v_dbkey='

save_dir = './'

for key,name in zip(db_keys,name):
    for (start_date,end_date) in date_lists:
        datetime_t = datetime%(start_date,end_date)

        url = base_url+'&'+ datetime_t +'&'+format+ '&'+ target + '&' + run_mode + '&' + js_flag + '&' +dbkey + key
        filename = save_dir + f"{name+"_"+start_date+"_"+end_date}.csv"  # Specify the name to save the file as

        # Send a GET request to the URL and stream the content
        response = requests.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print("Download complete!")
        else:
            print("Failed to download the file.")
