# Dataset
This folder is reserved for data sets. Please download the dataset from [Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FTU5UXE&version=DRAFT).

## Interpretation
The structure of the dataset folder:
```
--dataset_folder
  ---WATER
  ---WELL
  ---PUMP
  ---RAIN
  ---GATE
     ----S_0
     ----S_1
     ----S_2
     ----S_3
     ----S_4
     ----S_5
     ----S_6
     ----S_7
         -----COCO1_S # Station_name
              -------COCO1_S.csv   # data
              -------COCO1_S-loc_info.json  # station location
```

The header of csv files contains the following columns:  TIMESTAMP,VALUE,CONFIDENCE,INTERPOLATED_VALUE. 
- **TIMESTAMP**: the time of this row
- **VALUE**: the average result of collecting raw data by time 
- **CONFIDENCE**: the number of data point of the raw data
- **INTERPOLATED_VALUE**: the final data after interpolating VALUE

The interpretation of .json location file. We provide the visualization python script in /src/location_visualization.py
 - **Latitude**
 - **Longitude**
 - **X COORD**
 - **Y COORD**


## Other data
We also provide the three interest area selection json files(threeparts_[x]_map_locations_[split].json) and the file need by spatial ablation study(S6_map_locations_area[x].json). The [x] is a placholder of number, [split] is a placeholder of splits. 

We also provide an additional file of flood observation, FLOOD_OBSERVATION_REPOSITORY_V2024.csv.
