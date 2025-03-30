import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from pyproj import Proj, transform
import folium

florida_state_plane = Proj(init="epsg:2236")
wgs84 = Proj(init="epsg:4326")

def utm_to_latlon(easting, northing):
    lon, lat = transform(florida_state_plane, wgs84, easting, northing)
    return lat, lon

def main(num=0,dir = '../dataset'):
    x_str = 'X COORD'
    y_str = 'Y COORD'

    path_dir = f'{dir}/WATER/S_{num}'
    json_files = []
    for first_level in os.listdir(path_dir):
        first_level_path = os.path.join(path_dir, first_level)
        if os.path.isdir(first_level_path):
            for file in os.listdir(first_level_path):
                if file.endswith(".json"):
                    json_files.append(os.path.join(first_level_path, file))

    water_Latitude = []
    water_Longitude = []
    for json_file in json_files:
        with open(json_file) as fp:
            dd = json.load(fp)
        water_Latitude.append(dd[x_str])
        water_Longitude.append(dd[y_str])

    path_dir = f'{dir}/WELL/S_{num}'
    json_files = []
    for first_level in os.listdir(path_dir):
        first_level_path = os.path.join(path_dir, first_level)
        if os.path.isdir(first_level_path):
            for file in os.listdir(first_level_path):
                if file.endswith(".json"):
                    json_files.append(os.path.join(first_level_path, file))

    well_Latitude = []
    well_Longitude = []
    for json_file in json_files:
        with open(json_file) as fp:
            dd = json.load(fp)
        well_Latitude.append(dd[x_str])
        well_Longitude.append(dd[y_str])

    path_dir = f'{dir}/RAIN/S_{num}'
    json_files = []
    for first_level in os.listdir(path_dir):
        first_level_path = os.path.join(path_dir, first_level)
        if os.path.isdir(first_level_path):
            for file in os.listdir(first_level_path):
                if file.endswith(".json"):
                    json_files.append(os.path.join(first_level_path, file))

    rain_Latitude = []
    rain_Longitude = []
    for json_file in json_files:
        with open(json_file) as fp:
            dd = json.load(fp)
        rain_Latitude.append(dd[x_str])
        rain_Longitude.append(dd[y_str])

    path_dir = f'{dir}/PUMP/S_{num}'
    json_files = []
    pump_csvx_files = []
    for first_level in os.listdir(path_dir):
        first_level_path = os.path.join(path_dir, first_level)
        if os.path.isdir(first_level_path):
            for file in os.listdir(first_level_path):
                if file.endswith(".json"):
                    json_files.append(os.path.join(first_level_path, file))

    pump_Latitude = []
    pump_Longitude = []
    for json_file in json_files:
        with open(json_file) as fp:
            dd = json.load(fp)
        pump_Latitude.append(dd[x_str])
        pump_Longitude.append(dd[y_str])

    path_dir = f'{dir}/GATE/S_{num}'
    json_files = []
    for first_level in os.listdir(path_dir):
        first_level_path = os.path.join(path_dir, first_level)
        if os.path.isdir(first_level_path):
            for file in os.listdir(first_level_path):
                if file.endswith(".json"):
                    json_files.append(os.path.join(first_level_path, file))

    gate_Latitude = []
    gate_Longitude = []
    for json_file in json_files:
        with open(json_file) as fp:
            dd = json.load(fp)
        gate_Latitude.append(dd[x_str])
        gate_Longitude.append(dd[y_str])

    group_colors = {
        "WATER": "blue",
        "WELL": "green",
        "RAIN": "red",
        "PUMP": "purple",
        "GATE": "orange"
    }

    group_data = {
        "WATER": {
            "latitudes": water_Latitude,  # [25.7617, 26.1224],
            "longitudes": water_Longitude  # [-80.1918, -80.1373]
        },
        "WELL": {
            "latitudes": well_Latitude,  # [40.7128, 40.7306],
            "longitudes": well_Longitude  # [-74.0060, -73.9352]
        },
        "RAIN": {
            "latitudes": rain_Latitude,  # [34.0522, 34.0736],
            "longitudes": rain_Longitude  # [-118.2437, -118.4004]
        },
        "PUMP": {
            "latitudes": pump_Latitude,  # [41.8781, 41.8500],
            "longitudes": pump_Longitude  # [-87.6298, -87.6501]
        },
        "GATE": {
            "latitudes": gate_Latitude,  # [37.7749, 37.8044],
            "longitudes": gate_Longitude  # [-122.4194, -122.2711]
        }
    }

    m = folium.Map(location=[27.5, -81.5], zoom_start=7)
    # m = folium.Map(location=[38.0, -97.0], zoom_start=4)
    # Add markers for each group
    for group, data in group_data.items():
        for lat, lon in zip(data["latitudes"], data["longitudes"]):
            lat, lon = utm_to_latlon(lat, lon)
            folium.Marker(
                location=[lat, lon],
                popup=f"{group} ({lat}, {lon})",
                icon=folium.Icon(color=group_colors.get(group, "gray"))
            ).add_to(m)

    # Save and display the map
    m.save(f"./map_locations_{num}.html")

if __name__=="__main__":
    for i in range(8):
        main(i)

