import streamlit as st
import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
from sklearn.cluster import KMeans
import stackstac
import planetary_computer
from pystac_client import Client
from matplotlib.patches import Patch
import warnings
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

warnings.filterwarnings("ignore", category=UserWarning)

# Sitzung mit Retry-Mechanismus für stabile API-Abfragen
session = requests.Session()
retry = Retry(connect=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)


def heatmap_mit_temperaturdifferenzen(ort_name, jahr=2022, radius_km=1.5, resolution_km=1.0):
    geolocator = Nominatim(user_agent="hitze-check")
    try:
        location = geolocator.geocode(ort_name, timeout=10)
    except Exception as e:
        st.error(f"🌍 Geokodierung fehlgeschlagen: {e}")
        return None

    if not location:
        st.warning("❗ Ort konnte nicht gefunden werden.")
        return None

    lat0, lon0 = location.latitude, location.longitude
    lats = np.arange(lat0 - radius_km / 111, lat0 + radius_km / 111, resolution_km / 111)
    lons = np.arange(lon0 - radius_km / 85, lon0 + radius_km / 85, resolution_km / 85)

    punkt_daten = []
    ref_temp = None

    for lat in lats:
        for lon in lons:
            try:
                url = (
                    f"https://archive-api.open-meteo.com/v1/archive?"
                    f"latitude={lat}&longitude={lon}"
                    f"&start_date={jahr}-06-01&end_date={jahr}-08-31"
                    f"&daily=temperature_2m_max&timezone=auto"
                )
                r = session.get(url, timeout=5)
                if r.status_code != 200:
                    continue
                temps = r.json().get("daily", {}).get("temperature_2m_max", [])
                if not temps:
                    continue
                avg_temp = round(np.mean(temps), 2)

                punkt_daten.append([lat, lon, avg_temp])

                if abs(lat - lat0) < resolution_km / 222 and abs(lon - lon0) < resolution_km / 170:
                    ref_temp = avg_temp

            except Exception:
                continue

    if not punkt_daten or ref_temp is None:
        st.warning("⚠️ Nicht genug Temperaturdaten oder Mittelpunktwert nicht verfügbar.")
        return None

    differenzpunkte = [
        [lat, lon, round(temp - ref_temp, 2)]
        for lat, lon, temp in punkt_daten
    ]

    m = folium.Map(location=[lat0, lon0], zoom_start=13, tiles="CartoDB positron")
    HeatMap(
        [[lat, lon, abs(diff)] for lat, lon, diff in differenzpunkte],
        radius=18,
        blur=25,
        max_zoom=13,
        gradient={0.0: "green", 0.3: "lightyellow", 0.6: "orange", 1.0: "red"}
    ).add_to(m)

    for lat, lon, diff in differenzpunkte:
        sign = "+" if diff > 0 else ("−" if diff < 0 else "±")
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=f"<div style='font-size:10pt; color:black'><b>{sign}{abs(diff):.2f}°C</b></div>")
        ).add_to(m)

    return m
