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

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(layout="wide")
st.title("🌿 Urban Greening Prototype")

stadtname = st.text_input("🏙️ Gib den Namen des Stadtteils ein:", "Maxvorstadt, München")
baumdateiname = "Baumliste_neu (2).csv"

def lade_temperaturverlauf(ort_name, tage=7):
    ort_gdf = ox.geocode_to_gdf(ort_name)
    center = ort_gdf.geometry.iloc[0].centroid
    lat, lon = center.y, center.x

    start_date = (datetime.utcnow() - timedelta(days=tage)).date()
    end_date = datetime.utcnow().date()

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m&timezone=Europe%2FBerlin"
    )

    r = requests.get(url)
    if r.status_code != 200:
        st.error("Temperaturdaten konnten nicht abgerufen werden.")
        return None

    data = r.json()
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"]
    })
    return df

if st.button("🔍 Analyse starten"):
    with st.spinner("Lade Gebiet und Daten..."):
        gebiet = ox.geocode_to_gdf(stadtname)
        polygon = gebiet.geometry.iloc[0]
        print("Gebiet geladen:", polygon.bounds)
        utm_crs = gebiet.estimate_utm_crs()
        gebiet = gebiet.to_crs(utm_crs)
        area = gebiet.geometry.iloc[0].buffer(0)

        tags_buildings = {"building": True}
        tags_green = {
            "leisure": ["park", "garden"],
            "landuse": ["grass", "meadow", "forest"],
            "natural": ["wood", "tree_row", "scrub"]
        }
        buildings = ox.features_from_polygon(polygon, tags=tags_buildings).to_crs(utm_crs)
        greens = ox.features_from_polygon(polygon, tags=tags_green).to_crs(utm_crs)
        buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty]
        greens = greens[greens.geometry.is_valid & ~greens.geometry.is_empty]

        print("Gebäudeanzahl:", len(buildings))
        print("Grünflächenanzahl:", len(greens))

        if buildings.empty:
            st.error("⚠️ Keine Gebäudedaten gefunden. Bitte einen allgemeineren Ort wählen.")
            st.stop()

        streets = ox.graph_from_polygon(polygon, network_type="walk")
        edges = ox.graph_to_gdfs(streets, nodes=False).to_crs(utm_crs)
        edges = edges[edges.geometry.type == "LineString"]

        cell_size = 50
        minx, miny, maxx, maxy = area.bounds
        grid_cells = [
            box(x, y, x + cell_size, y + cell_size)
            for x in np.arange(minx, maxx, cell_size)
            for y in np.arange(miny, maxy, cell_size)
            if box(x, y, x + cell_size, y + cell_size).intersects(area)
        ]
        grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)

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

        max_dist = 500
        grid["score_density_norm"] = grid["score_density"]
        grid["score_distance_norm"] = 1 - np.clip(grid["dist_to_green"] / max_dist, 0, 1)
        grid["score_total"] = 0.5 * grid["score_density_norm"] + 0.5 * grid["score_distance_norm"]

    st.subheader("🌡 Temperaturverlauf der letzten 7 Tage")
    temp_df = lade_temperaturverlauf(stadtname)
    if temp_df is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        temp_df.plot(x="time", y="temperature", ax=ax, legend=False)
        ax.set_ylabel("°C")
        ax.set_title(f"Temperaturverlauf in {stadtname}")
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("1️⃣ Gebäudedichte (Rot = dicht bebaut)")
    fig, ax = plt.subplots(figsize=(8, 8))
    grid.plot(ax=ax, column="building_ratio", cmap="Reds", legend=True, edgecolor="grey", linewidth=0.2)
    buildings.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)
    gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
    ax.axis("equal")
    ax.set_title("Gebäudedichte")
    st.pyplot(fig)

    st.subheader("2️⃣ Distanz zu Grünflächen")
    fig, ax = plt.subplots(figsize=(8, 8))
    grid.plot(ax=ax, column="score_distance_norm", cmap="Reds", legend=True, edgecolor="grey", linewidth=0.2)
    greens.plot(ax=ax, color="green", alpha=0.3, edgecolor="darkgreen")
    gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
    ax.axis("equal")
    ax.set_title("Distanz zu Grünflächen")
    st.pyplot(fig)

    st.subheader("3️⃣ Kombiniertes Begrünungspotenzial")
    fig, ax = plt.subplots(figsize=(8, 8))
    grid.plot(ax=ax, column="score_total", cmap="YlGn", legend=True, edgecolor="grey", linewidth=0.2)
    buildings.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)
    greens.plot(ax=ax, color="green", alpha=0.3, edgecolor="darkgreen")
    gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
    ax.axis("equal")
    ax.set_title("Kombiniertes Begrünungspotenzial")
    st.pyplot(fig)

    st.subheader("🌳 Baumempfehlung")
    baumdaten = pd.read_csv(baumdateiname, sep=";")
    baumdaten.columns = [c.strip().lower() for c in baumdaten.columns]

    def finde_baumempfehlung(line, buffer_dist=15):
        buffer = line.buffer(buffer_dist)
        relevant = grid[grid.intersects(buffer)]
        score = relevant["score_total"].mean() if not relevant.empty else np.nan
        if np.isnan(score):
            return pd.Series([np.nan, None])
        df = baumdaten.copy()
        df["punkte"] = 0
        if "stadtklimafest" in df.columns:
            df["punkte"] += df["stadtklimafest"].fillna(False).astype(int) * 2
        if "trockenheitsverträglich" in df.columns:
            df["punkte"] += df["trockenheitsverträglich"].fillna(False).astype(int)
        if "salzverträglich" in df.columns:
            df["punkte"] += df["salzverträglich"].fillna(False).astype(int)
        if "wurzeldruckverträglich" in df.columns:
            df["punkte"] += df["wurzeldruckverträglich"].fillna(False).astype(int)
        if "maximale_endhöhe" in df.columns:
            df["punkte"] += (df["maximale_endhöhe"] <= 15).fillna(False).astype(int)
        if "ph_toleranz_min" in df.columns and "ph_toleranz_max" in df.columns:
            df["punkte"] += ((df["ph_toleranz_min"] <= 7.5) & (df["ph_toleranz_max"] >= 7.5)).fillna(False).astype(int)
        if "standort_geeignet" in df.columns:
            df["punkte"] += df["standort_geeignet"].fillna("").str.contains("trocken", case=False).astype(int)
        if score > 0.7:
            df["punkte"] += df["wuchsform"].fillna("").str.contains("schmal|säule", case=False).astype(int)
        elif score > 0.4:
            df["punkte"] += df["kronenform"].fillna("").str.contains("rundlich|eiförmig", case=False).astype(int)
        df = df[df["punkte"] >= 3]
        if df.empty:
            return pd.Series([score, None])
        max_punkte = df["punkte"].max()
        df = df[df["punkte"] >= max_punkte * 0.75]
        top_arten = df.sort_values("punkte", ascending=False)["baumart"].dropna().unique().tolist()
        return pd.Series([score, ", ".join(top_arten[:5])])

    st.info("Dies kann je nach Gebiet einige Sekunden dauern...")
    edges[["begruenungspotenzial", "baumempfehlung"]] = edges.geometry.apply(finde_baumempfehlung)

    st.subheader("4️⃣ Begrünungspotenzial an Straßen")
    fig, ax = plt.subplots(figsize=(8, 8))
    edges.plot(ax=ax, column="begruenungspotenzial", cmap="Blues", legend=True, linewidth=2)
    gebiet.boundary.plot(ax=ax, color="black")
    ax.axis("equal")
    ax.set_title("Straßenbegrünung")
    st.pyplot(fig)

    st.subheader("🔎 Empfehlungstabelle")
    st.dataframe(edges[["name", "begruenungspotenzial", "baumempfehlung"]].dropna().reset_index(drop=True))
