import frigis as fg
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

def main():
    stadtteil = fg.input.text("🏙️ Stadtteilname eingeben", default="Maxvorstadt, München")
    fg.notify("📍 Lade Gebiet...")
    gebiet = ox.geocode_to_gdf(stadtteil)
    polygon = gebiet.geometry.iloc[0]
    utm_crs = gebiet.estimate_utm_crs()
    gebiet = gebiet.to_crs(utm_crs)
    area = gebiet.geometry.iloc[0].buffer(0)

    # Gebäude und Grünflächen laden
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

    # Raster erstellen
    cell_size = 50
    minx, miny, maxx, maxy = area.bounds
    grid_cells = [
        box(x, y, x + cell_size, y + cell_size)
        for x in np.arange(minx, maxx, cell_size)
        for y in np.arange(miny, maxy, cell_size)
        if box(x, y, x + cell_size, y + cell_size).intersects(area)
    ]
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)

    # Gebäudedichte
    fg.notify("🏗️ Analysiere Gebäudedichte...")
    def calc_building_ratio(cell):
        intersecting = buildings[buildings.intersects(cell)]
        return intersecting.intersection(cell).area.sum() / cell.area if not intersecting.empty else 0

    grid["building_ratio"] = grid.geometry.apply(calc_building_ratio)
    fg.plot(grid.plot(column="building_ratio", cmap="Reds", edgecolor="grey", linewidth=0.2, legend=True).figure)

    # Distanz zu Grünflächen
    fg.notify("🌳 Berechne Grünflächendistanz...")
    def safe_distance(gdf, geom):
        if gdf.empty or geom.is_empty:
            return np.nan
        return gdf.geometry.distance(geom.centroid).min()

    grid["dist_to_green"] = grid.geometry.apply(lambda g: safe_distance(greens, g))
    grid["score_distance_norm"] = np.clip(grid["dist_to_green"] / 500, 0, 1)
    fg.plot(grid.plot(column="score_distance_norm", cmap="Reds", edgecolor="grey", linewidth=0.2, legend=True).figure)

    # Temperatur Heatmap
    fg.notify("🌡️ Lade Temperaturdaten...")
    geolocator = Nominatim(user_agent="frigis-app")
    location = geolocator.geocode(stadtteil)
    if location:
        lat0, lon0 = location.latitude, location.longitude
        lats = np.arange(lat0 - 3/111, lat0 + 3/111, 1.0/111)
        lons = np.arange(lon0 - 3/85, lon0 + 3/85, 1.0/85)
        points = []
        for lat in lats:
            for lon in lons:
                url = (
                    f"https://archive-api.open-meteo.com/v1/archive?"
                    f"latitude={lat}&longitude={lon}"
                    f"&start_date=2022-06-01&end_date=2022-08-31"
                    f"&daily=temperature_2m_max&timezone=auto"
                )
                r = requests.get(url)
                if r.status_code != 200:
                    continue
                temps = r.json().get("daily", {}).get("temperature_2m_max", [])
                if temps:
                    avg_temp = round(np.mean(temps), 2)
                    if avg_temp >= 20.0:
                        points.append([lat, lon, avg_temp])
        if points:
            m = folium.Map(location=[lat0, lon0], zoom_start=13, tiles="CartoDB positron")
            HeatMap([[p[0], p[1], p[2]] for p in points], radius=18, blur=25).add_to(m)
            fg.map(m)

    # Reflektivität
    fg.notify("🛰️ Analysiere Reflektivität mit Sentinel-2...")
    ort = ox.geocode_to_gdf(stadtteil)
    bbox = ort.total_bounds
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox.tolist(),
        datetime="2020-01-01/2024-12-31",
        query={"eo:cloud_cover": {"lt": 5}}
    )
    items = list(search.get_items())
    if items:
        item = planetary_computer.sign(items[0])
        stack = stackstac.stack([item], assets=["B04", "B03", "B02"], resolution=10, bounds_latlon=bbox.tolist(), epsg=32632)
        rgb = stack.isel(band=[0,1,2], time=0).transpose("y","x","band").values
        rgb = np.nan_to_num(rgb)
        rgb_scaled = np.clip((rgb / 3000) * 255, 0, 255).astype(np.uint8)

        h, w, _ = rgb_scaled.shape
        pixels = rgb_scaled.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42).fit(pixels)
        labels = kmeans.labels_
        gray_values = np.linspace(0, 255, 5).astype(int)
        gray_colors = np.stack([gray_values]*3, axis=1)
        cluster_image = gray_colors[labels].reshape(h, w, 3).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(cluster_image)
        ax.axis("off")
        fg.plot(fig)

fg.run(main)
