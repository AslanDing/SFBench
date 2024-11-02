import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin


def html_table_to_dataframe(text):
    soup = BeautifulSoup(text, 'lxml')

    # Find the table
    res = soup.find('form')
    table = res.find('table')
    # table = soup.find('table')

    # Extract the header
    headers = []
    for th in table.find_all('th'):
        headers.append(th.get_text(strip=True))
    headers.append('start_date')
    headers.append('end_date')

    # Extract the rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip the header row
        cols = tr.find_all('td')
        row_data = [col.get_text(strip=True) for col in cols]

        # Append hidden inputs for start and end dates
        hidden_inputs = tr.find_all('input', type='hidden')
        for hidden in hidden_inputs:
            row_data.append(hidden['value'])  # Add the hidden input value

        if len(row_data)<27:
            continue
        else:
            rows.append(row_data)

    return rows , headers
    # Create DataFrame
    # df = pd.DataFrame(rows, columns=headers)
    # return df

def get_next_link(text):
    # all_links = []
    soup = BeautifulSoup(text, 'html.parser')
    # target_on_click = "passDbkeys(dbkeyList)\""
    # for link in soup.find_all('a', onClick=target_on_click, href=True):
    #     all_links.append(link['href'])
    hrefs = []
    target_img_src = "/img/forward.gif"
    for a in soup.find_all('a'):
        img = a.find('img')
        if img and img.get('src') == target_img_src:  # Check if the <img> exists and matches the src
            hrefs.append(a['href'])  # Collect the href
    return hrefs

def get_numbers(text):
    soup = BeautifulSoup(text, 'html.parser')
    record_text = soup.find(text=re.compile("Query returned.*record"))
    if record_text:
        total_records = int(re.search(r"Query returned (\d+) record", record_text).group(1))
        print(f"Total records found: {total_records}")
        return total_records
    return -1

base_url = 'https://my.sfwmd.gov'
# from this site
# https://my.sfwmd.gov/dbhydroplsql/show_dbkey_info.web_qry_parameter?v_category=SW&v_js_flag=Y
start_url = 'https://my.sfwmd.gov/dbhydroplsql/show_dbkey_info.show_dbkeys_matched?v_js_flag=Y&v_category=SW&v_station=&v_site=&v_data_type=STG&v_dbkey_list_flag=Y&v_order_by=STATION'
# start_url = 'https://my.sfwmd.gov/dbhydroplsql/show_dbkey_info.show_dbkeys_matched?v_station=WPBC&v_js_flag=Y'
response = requests.get(start_url)

headers = []
rows = []
rowx, header = html_table_to_dataframe(response.text)
next_all_links = get_next_link(response.text)
rows.extend(rowx)
headers.extend(header)
total_number = get_numbers(response.text)
print("total number: ", total_number)

for start_int in range(100,total_number,100):
    url = f"/dbhydroplsql/show_dbkey_info.show_dbkeys_matched?display_start={start_int+1}&display_quantity=100&v_start_date=&v_end_date=&v_order_by=STATION&v_store_dbkey=&v_js_flag=Y&v_data_type=STG"
    response = requests.get(base_url + url)
    rowx, header = html_table_to_dataframe(response.text)
    rows.extend(rowx)

# record_text = False
# while not record_text:
#     try:
#         response = requests.get(base_url+next_all_links[0])
#         rowx, header = html_table_to_dataframe(response.text)
#         rows.extend(rowx)
#         headers.extend(header)
#         next_all_links = get_next_link(response.text)
#         if len(next_all_links)<1:
#             print("no next")
#     except:
#         break

df = pd.DataFrame(rows, columns=headers)
print("table length, ", df.shape)
df.to_csv('./dbkey_tables.csv')



