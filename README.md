# WaterBenchmark
A water level dataset and benchmark

### Download the surface water data 

## get water moniter stations dbkey 
- use the [url](https://my.sfwmd.gov/dbhydroplsql/show_dbkey_info.web_qry_parameter?v_category=SW&v_js_flag=Y) to get the api fliter parameters
- use /dataset_download/download_tables.py download the csv file(contains dbkey,data type, start date, end date, locations)
- use /dataset_download/download_dbkey.py download the csv file according to the table

## preprocess
- calculate the start date and end date(select water stations)
- calculate the hourly data(missing data refilling)

