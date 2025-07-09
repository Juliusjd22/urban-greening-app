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
warnings.filterwarnings("ignore", category=UserWarning)

def gebaeudedichte_analysieren_und_plotten(grid, buildings, gebiet):
    def calc_building_ratio(cell):
        intersecting = buildings[buildings.intersects(cell)]
        return intersecting.intersection(cell).area.sum() / cell.area if not intersecting.empty else 0

    grid["building_ratio"] = grid.geometry.apply(calc_building_ratio)

    fig, ax = plt.subplots(figsize=(8, 8))
    grid.plot(ax=ax, column="building_ratio", cmap="Reds", legend=True,
              edgecolor="grey", linewidth=0.2)
    buildings.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)
    gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
    ax.set_title("1️⃣ Gebäudedichte (Rot = dicht bebaut)")
    ax.axis("equal")
    plt.tight_layout()
    return fig

def distanz_zu_gruenflaechen_analysieren_und_plotten(grid, greens, gebiet, max_dist=500):
    def safe_distance(gdf, geom):
        if gdf.empty or geom.is_empty:
            return np.nan
        return gdf.geometry.distance(geom.centroid).min()

    grid["dist_to_green"] = grid.geometry.apply(lambda g: safe_distance(greens, g))
    grid["score_distance_norm"] = np.clip(grid["dist_to_green"] / max_dist, 0, 1)

    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(8, 8))
    grid.plot(ax=ax, column="score_distance_norm", cmap=cmap, norm=norm,
              edgecolor="grey", linewidth=0.2, legend=True,
              legend_kwds={"label": "Entfernung zu Grün (Rot = weit entfernt)"})
    greens.plot(ax=ax, color="green", alpha=0.5, edgecolor="darkgreen")
    gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
    ax.set_title("2️⃣ Distanz zu Grünflächen (Rot = weit entfernt, Grün = vorhanden)")
    ax.axis("equal")
    plt.tight_layout()
    return fig

def analysiere_reflektivitaet_graustufen(stadtteil_name, n_clusters=5, year_range="2020-01-01/2024-12-31"):
    ort = ox.geocode_to_gdf(stadtteil_name)
    bbox = ort.total_bounds

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox.tolist(),
        datetime=year_range,
        query={"eo:cloud_cover": {"lt": 5}}
    )
    items = list(search.get_items())
    if not items:
        st.warning("❌ Kein geeignetes Sentinel-2 Bild gefunden.")
        return None

    item = planetary_computer.sign(items[0])
    utm_crs = ort.estimate_utm_crs().to_epsg()
    stack = stackstac.stack(
        [item],
        assets=["B04", "B03", "B02"],
        resolution=10,
        bounds_latlon=bbox.tolist(),
        epsg=utm_crs
    )
    rgb = stack.isel(band=[0,1,2], time=0).transpose("y","x","band").values
    rgb = np.nan_to_num(rgb)
    rgb_scaled = np.clip((rgb / 3000) * 255, 0, 255).astype(np.uint8)

    h, w, _ = rgb_scaled.shape
    pixels = rgb_scaled.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
    labels = kmeans.labels_

    cluster_info = []
    for i in range(n_clusters):
        cluster_pixels = pixels[labels == i]
        if len(cluster_pixels) == 0:
            cluster_info.append((i, 0, "Keine Daten"))
            continue
        helligkeit = cluster_pixels.mean(axis=1).mean() / 255
        beschreibung = (
            "🌞 Sehr hell (hohe Reflektivität)" if helligkeit > 0.75 else
            "🔆 Hell (moderat reflektierend)" if helligkeit > 0.5 else
            "🌥️ Mittel (neutral)" if helligkeit > 0.35 else
            "🌡️ Dunkel (hohes Aufheizungspotenzial)"
        )
        cluster_info.append((i, round(helligkeit, 2), beschreibung))

    gray_values = np.linspace(0, 255, n_clusters).astype(int)
    gray_colors = np.stack([gray_values]*3, axis=1)
    cluster_image = gray_colors[labels].reshape(h, w, 3).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(cluster_image)
    ax.axis("off")

    legend_elements = [
        Patch(facecolor=gray_colors[i]/255, edgecolor='black',
              label=f"Cluster {i}: {cluster_info[i][2]} ({cluster_info[i][1]*100:.0f}%)")
        for i in range(n_clusters)
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=1, frameon=True, fontsize="small")
    plt.tight_layout()
    return fig

def main():
    st.title("🌿 friGIS")

    st.markdown("""
        by Philippa, Samuel, Julius  
        Hey, sehr cool, dass du unseren Prototypen nutzt. Dieser Prototyp soll zeigen, 
        auf Basis welcher Daten wir ...
    """)

    stadtteil = st.text_input("🏙️ Stadtteilname eingeben", value="Maxvorstadt, München")
    starten = st.button("🔍 Analyse starten")

    if not starten or not stadtteil:
        return

    try:
        gebiet = ox.geocode_to_gdf(stadtteil)
    except Exception as e:
        st.error(f"📍 Gebiet konnte nicht geladen werden: {e}")
        return

    polygon = gebiet.geometry.iloc[0]
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

    cell_size = 50
    minx, miny, maxx, maxy = area.bounds
    grid_cells = [
        box(x, y, x + cell_size, y + cell_size)
        for x in np.arange(minx, maxx, cell_size)
        for y in np.arange(miny, maxy, cell_size)
        if box(x, y, x + cell_size, y + cell_size).intersects(area)
    ]
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)

    st.subheader("Gebäudedichte")
    fig1 = gebaeudedichte_analysieren_und_plotten(grid, buildings, gebiet)
    st.pyplot(fig1)

    st.subheader("Distanz zu Grünflächen")
    fig2 = distanz_zu_gruenflaechen_analysieren_und_plotten(grid, greens, gebiet)
    st.pyplot(fig2)

    st.subheader("Temperatur Heatmap mit Temperaturwerten")
    heatmap = heatmap_mit_temperaturlabels(ort_name=stadtteil)
    if heatmap:
        st.components.v1.html(heatmap._repr_html_(), height=600)
    else:
        st.warning("Keine Temperaturdaten gefunden.")

    st.subheader("k-Means Clusteranalyse von Satellitendaten")
    fig3 = analysiere_reflektivitaet_graustufen(stadtteil, n_clusters=5)
    if fig3:
        st.pyplot(fig3)

if __name__ == "__main__":
    main()
