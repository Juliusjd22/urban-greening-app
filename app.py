
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import box
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import requests
import warnings
import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(layout="wide")
st.title("🌿 Urban Greening Prototype")

# ------------------------- FUNKTIONEN -------------------------

def lade_geometrie(ort):
    gdf = ox.geocode_to_gdf(ort)
    return gdf, gdf.geometry.iloc[0].centroid, gdf.estimate_utm_crs()

def lade_gebaeude_und_gruenflaechen(polygon, utm_crs):
    tags_buildings = {"building": True}
    tags_green = {
        "leisure": ["park", "garden"],
        "landuse": ["grass", "meadow", "forest"],
        "natural": ["wood", "tree_row", "scrub"]
    }
    buildings = ox.features_from_polygon(polygon, tags=tags_buildings).to_crs(utm_crs)
    greens = ox.features_from_polygon(polygon, tags=tags_green).to_crs(utm_crs)
    return buildings, greens

def rasterisiere(gebiet, cell_size=50):
    area = gebiet.geometry.iloc[0].buffer(0)
    minx, miny, maxx, maxy = area.bounds
    grid_cells = [box(x, y, x + cell_size, y + cell_size)
                  for x in np.arange(minx, maxx, cell_size)
                  for y in np.arange(miny, maxy, cell_size)
                  if box(x, y, x + cell_size, y + cell_size).intersects(area)]
    return gpd.GeoDataFrame({'geometry': grid_cells}, crs=gebiet.crs)

def berechne_scores(grid, buildings, greens):
    def calc_building_ratio(cell):
        intersecting = buildings[buildings.intersects(cell)]
        return intersecting.intersection(cell).area.sum() / cell.area if not intersecting.empty else 0

    def safe_distance(gdf, geom):
        if gdf.empty or geom.is_empty:
            return np.nan
        return gdf.geometry.distance(geom.centroid).min()

    grid["building_ratio"] = grid.geometry.apply(calc_building_ratio)
    grid["dist_to_green"] = grid.geometry.apply(lambda g: safe_distance(greens, g))
    grid["score_density"] = 1 - grid["building_ratio"]
    grid["score_density_norm"] = grid["score_density"]
    grid["score_distance_norm"] = 1 - np.clip(grid["dist_to_green"] / 500, 0, 1)
    grid["score_total"] = 0.5 * grid["score_density_norm"] + 0.5 * grid["score_distance_norm"]
    return grid

def lade_temperaturverlauf(lat, lon, tage=7):
    start_date = (datetime.utcnow() - timedelta(days=tage)).date()
    end_date = datetime.utcnow().date()
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m&timezone=Europe%2FBerlin"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    return pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"]
    })

def lade_messstationen(lat, lon):
    url = f"https://api.open-meteo.com/v1/stations?latitude={lat}&longitude={lon}&distance=15000"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    stations = r.json().get("results", [])
    daten = []
    for s in stations:
        temps = s.get("temperature_2m_mean")
        if temps:
            daten.append({
                "id": s.get("id"),
                "name": s.get("name"),
                "lat": s.get("latitude"),
                "lon": s.get("longitude"),
                "avg_temp": temps.get("value"),
                "std_temp": temps.get("standard_deviation")
            })
    return pd.DataFrame(daten)

def zeige_temperaturverlauf(df, ort):
    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot(x="time", y="temperature", ax=ax, legend=False)
    ax.set_ylabel("°C")
    ax.set_title(f"Temperaturverlauf in {ort}")
    ax.grid(True)
    st.pyplot(fig)

def zeige_messstationen(df):
    st.dataframe(df)
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=12)
    for _, row in df.iterrows():
        popup = f"<b>{row['name']}</b><br>Ø Temperatur: {row['avg_temp']} °C<br>σ: {row['std_temp']}"
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=6,
            fill=True,
            color="blue",
            fill_color="blue",
            fill_opacity=0.7,
            popup=popup
        ).add_to(m)
    st_folium(m, width=800, height=500)

# ------------------------- HAUPTANWENDUNG -------------------------

stadtname = st.text_input("🏙️ Gib den Namen des Stadtteils ein:", "Maxvorstadt, München")

if st.button("🔍 Analyse starten"):
    st.subheader("📍 Lade Geodaten...")
    ort_gdf, zentrum, utm_crs = lade_geometrie(stadtname)
    ort_gdf = ort_gdf.to_crs(utm_crs)
    polygon = ort_gdf.geometry.iloc[0]
    buildings, greens = lade_gebaeude_und_gruenflaechen(polygon, utm_crs)
    grid = rasterisiere(ort_gdf)
    grid = berechne_scores(grid, buildings, greens)

    st.subheader("🌡 Temperaturverlauf")
    temp_df = lade_temperaturverlauf(zentrum.y, zentrum.x)
    if temp_df is not None:
        zeige_temperaturverlauf(temp_df, stadtname)

    st.subheader("📊 Messstationen im Umkreis")
    stations_df = lade_messstationen(zentrum.y, zentrum.x)
    if stations_df is not None:
        zeige_messstationen(stations_df)
