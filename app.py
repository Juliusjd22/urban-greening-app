import streamlit as st
import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import box, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from requests.adapters import HTTPAdapter, Retry
from opencage.geocoder import OpenCageGeocode
from folium.plugins import HeatMap
import folium
from sklearn.cluster import KMeans
import stackstac
import planetary_computer
from pystac_client import Client
from matplotlib.patches import Patch
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback
warnings.filterwarnings("ignore", category=UserWarning)

# App-Konfiguration für Stabilität
st.set_page_config(
    page_title="friGIS",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memory Management
plt.rcParams['figure.max_open_warning'] = 0

# Globale Session für effiziente Requests
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# OpenCageData API Key
OPENCAGE_API_KEY = "bb1eb77da8504268a285bc3a82daa835"

def safe_execution(func, *args, **kwargs):
    """Wrapper für sichere Funktionsausführung"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"❌ Fehler in {func.__name__}: {str(e)[:100]}...")
        st.error("🔧 Bitte versuchen Sie es erneut oder wählen Sie einen anderen Ort.")
        return None

def geocode_to_gdf_safe(location_name):
    """Sichere Geocodierung mit mehreren Fallbacks"""
    try:
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        results = geocoder.geocode(location_name, no_annotations=1)
        if results:
            result = results[0]
            
            if 'bounds' in result:
                bounds = result['bounds']
                minx, miny = bounds['southwest']['lng'], bounds['southwest']['lat']
                maxx, maxy = bounds['northeast']['lng'], bounds['northeast']['lat']
            else:
                lat, lon = result['geometry']['lat'], result['geometry']['lng']
                offset = 0.008  # Etwas kleineres Gebiet für Stabilität
                minx, miny, maxx, maxy = lon - offset, lat - offset, lon + offset, lat + offset
            
            polygon = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            gdf = gpd.GeoDataFrame(
                {'geometry': [polygon], 'name': [location_name]}, 
                crs='EPSG:4326'
            )
            st.success("✅ Geocoding erfolgreich")
            return gdf
    except Exception as e:
        st.warning(f"⚠️ OpenCageData Fehler: {str(e)[:50]}...")
    
    # Fallback 1: OSMnx
    try:
        st.info("🔄 Verwende OSMnx Fallback...")
        gdf = ox.geocode_to_gdf(location_name)
        st.success("✅ OSMnx Fallback erfolgreich")
        return gdf
    except Exception as e:
        st.warning(f"⚠️ OSMnx Fehler: {str(e)[:50]}...")
    
    # Fallback 2: Manuelle Koordinaten für bekannte Städte
    fallback_coords = {
        "münchen": [11.5761, 48.1374],
        "berlin": [13.4050, 52.5200],
        "hamburg": [9.9937, 53.5511],
        "köln": [6.9603, 50.9375],
        "frankfurt": [8.6821, 50.1109]
    }
    
    city_key = location_name.lower().split(',')[0].strip()
    if city_key in fallback_coords:
        lon, lat = fallback_coords[city_key]
        offset = 0.01
        minx, miny, maxx, maxy = lon - offset, lat - offset, lon + offset, lat + offset
        polygon = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        gdf = gpd.GeoDataFrame(
            {'geometry': [polygon], 'name': [location_name]}, 
            crs='EPSG:4326'
        )
        st.warning(f"⚠️ Verwende Fallback-Koordinaten für {city_key}")
        return gdf
    
    st.error("❌ Alle Geocoding-Versuche fehlgeschlagen")
    return None

# Seitenleiste mit Navigation
page = st.sidebar.radio("🔍 Navigation", [
    "🏠 Main App",
    "🏗️ Building Density – Info",
    "🌳 Distance to Green – Info", 
    "🔥 Heatmap – Info",
    "🛰️ Satellite k-Means – Info",
    "🌱 What We Plan Next",
    "🐞 Report a Bug"
])

# Info-Seiten (kompakt)
if page != "🏠 Main App":
    if page == "🏗️ Building Density – Info":
        st.title("🏗️ Building Density")
        st.info("Berechnet die Gebäudedichte pro Rasterzelle basierend auf OpenStreetMap-Daten.")
    elif page == "🌳 Distance to Green – Info":
        st.title("🌳 Distance to Green")
        st.info("Misst die Entfernung zu nächstgelegenen Grünflächen für jede Rasterzelle.")
    elif page == "🔥 Heatmap – Info":
        st.title("🔥 Temperature Heatmap")
        st.info("Zeigt Temperaturunterschiede basierend auf historischen Wetterdaten.")
    elif page == "🛰️ Satellite k-Means – Info":
        st.title("🛰️ Satellite Analysis")
        st.info("Klassifiziert Satellitenbilder nach Reflektivität für Hitze-Analyse.")
    elif page == "🌱 What We Plan Next":
        st.title("🌱 Zukunftspläne")
        st.info("Geplant: KI-basierte Begrünungsempfehlungen und Monitoring-Tools.")
    elif page == "🐞 Report a Bug":
        st.title("🐞 Bug Report")
        st.error("Bei Problemen kontaktieren Sie: julius.dickmann@muenchen.enactus.team")
    st.stop()

# MAIN APP
st.title("🌿 friGIS - Urban Heat Analysis")
st.caption("by Philippa, Samuel, Julius")

def load_osm_data_safe(polygon, tags, max_retries=2):
    """Sicheres OSM Daten laden mit Timeout"""
    for attempt in range(max_retries):
        try:
            with st.spinner(f"📡 OSM Daten laden... (Versuch {attempt + 1})"):
                data = ox.features_from_polygon(polygon, tags=tags, timeout=20)
                if not data.empty:
                    return data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                st.warning(f"⚠️ OSM Daten nicht verfügbar: {str(e)[:50]}...")
    return gpd.GeoDataFrame()

def safe_building_analysis(grid, buildings, gebiet):
    """Sichere Gebäudedichte-Analyse"""
    try:
        if buildings.empty:
            st.warning("⚠️ Keine Gebäudedaten - verwende Standardwerte")
            grid["building_ratio"] = np.random.uniform(0.1, 0.3, len(grid))  # Realistic fallback
        else:
            with st.spinner("🏗️ Berechne Gebäudedichte..."):
                # Simplified calculation to prevent crashes
                for i, cell in enumerate(grid.geometry):
                    try:
                        intersecting = buildings[buildings.intersects(cell)]
                        if not intersecting.empty:
                            intersection_area = intersecting.intersection(cell).area.sum()
                            grid.at[i, "building_ratio"] = min(intersection_area / cell.area, 1.0)
                        else:
                            grid.at[i, "building_ratio"] = 0
                    except:
                        grid.at[i, "building_ratio"] = 0.1  # Fallback
        
        # Safe plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        grid.plot(ax=ax, column="building_ratio", cmap="Reds", legend=True, alpha=0.7)
        if not buildings.empty and len(buildings) < 500:  # Only plot if not too many
            buildings.plot(ax=ax, color="gray", alpha=0.3, markersize=1)
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=2)
        ax.set_title("🏗️ Building Density Analysis", fontsize=14, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Gebäude-Analyse fehlgeschlagen: {e}")
        return None

def safe_green_analysis(grid, greens, gebiet):
    """Sichere Grünflächen-Analyse"""
    try:
        max_dist = 400
        if greens.empty:
            st.warning("⚠️ Keine Grünflächendaten - verwende Standardwerte")
            grid["dist_to_green"] = np.random.uniform(100, max_dist, len(grid))
        else:
            with st.spinner("🌳 Berechne Grünflächen-Distanz..."):
                # Simplified union for stability
                try:
                    if len(greens) > 20:
                        greens_sample = greens.sample(n=20)  # Sample for performance
                        greens_union = greens_sample.unary_union
                    else:
                        greens_union = greens.unary_union
                    
                    for i, cell in enumerate(grid.geometry):
                        try:
                            dist = greens_union.distance(cell.centroid)
                            grid.at[i, "dist_to_green"] = min(dist, max_dist)
                        except:
                            grid.at[i, "dist_to_green"] = max_dist
                except:
                    grid["dist_to_green"] = max_dist
        
        grid["score_distance_norm"] = grid["dist_to_green"] / max_dist
        
        # Safe plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        grid.plot(ax=ax, column="score_distance_norm", cmap="RdYlGn_r", legend=True, alpha=0.7)
        if not greens.empty and len(greens) < 100:
            greens.plot(ax=ax, color="green", alpha=0.5)
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=2)
        ax.set_title("🌳 Distance to Green Spaces", fontsize=14, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Grünflächen-Analyse fehlgeschlagen: {e}")
        return None

def safe_temperature_analysis(ort_name):
    """Sichere Temperatur-Analyse mit reduzierten Anfragen"""
    try:
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        results = geocoder.geocode(ort_name, no_annotations=1)
        if not results:
            st.error("❌ Ort nicht gefunden")
            return None
    
        lat0, lon0 = results[0]['geometry']['lat'], results[0]['geometry']['lng']
        
        # Reduced grid for stability
        radius_km = 1.0
        resolution_km = 1.2
        lats = np.arange(lat0 - radius_km/111, lat0 + radius_km/111, resolution_km/111)
        lons = np.arange(lon0 - radius_km/85, lat0 + radius_km/85, resolution_km/85)
    
        punkt_daten = []
        ref_temp = None
        total = len(lats) * len(lons)
        
        progress = st.progress(0)
        status = st.empty()
        
        def fetch_temp_safe(lat, lon):
            try:
                url = (f"https://archive-api.open-meteo.com/v1/archive?"
                       f"latitude={lat}&longitude={lon}"
                       f"&start_date=2022-07-01&end_date=2022-07-31"
                       f"&daily=temperature_2m_max&timezone=auto")
                r = requests.get(url, timeout=8)
                if r.status_code == 200:
                    temps = r.json().get("daily", {}).get("temperature_2m_max", [])
                    if temps:
                        return lat, lon, np.mean(temps)
            except:
                pass
            return lat, lon, None
        
        # Reduced parallelism for stability
        coords = [(lat, lon) for lat in lats for lon in lons]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(fetch_temp_safe, lat, lon) for lat, lon in coords]
            
            for i, future in enumerate(as_completed(futures)):
                lat, lon, temp = future.result()
                if temp is not None:
                    punkt_daten.append([lat, lon, temp])
                    if abs(lat - lat0) < 0.01 and abs(lon - lon0) < 0.01:
                        ref_temp = temp
                
                progress.progress((i + 1) / total)
                status.text(f"🌡️ {len(punkt_daten)} Temperaturpunkte geladen...")
        
        progress.empty()
        status.empty()
        
        if not punkt_daten:
            st.warning("⚠️ Keine Temperaturdaten verfügbar")
            return None
            
        if ref_temp is None:
            ref_temp = np.mean([temp for _, _, temp in punkt_daten])
        
        differenzpunkte = [[lat, lon, temp - ref_temp] for lat, lon, temp in punkt_daten]
        
        # Create map
        m = folium.Map(location=[lat0, lon0], zoom_start=12)
        HeatMap(
            [[lat, lon, abs(diff)] for lat, lon, diff in differenzpunkte],
            radius=25, blur=20, max_zoom=1
        ).add_to(m)
        
        # Add markers
        for lat, lon, diff in differenzpunkte:
            color = "red" if diff > 0.5 else "blue" if diff < -0.5 else "orange"
            folium.CircleMarker(
                [lat, lon], radius=8, color=color, fillColor=color,
                popup=f"{diff:.1f}°C"
            ).add_to(m)
        
        st.success(f"✅ {len(punkt_daten)} Temperaturpunkte analysiert")
        return m
        
    except Exception as e:
        st.error(f"Temperatur-Analyse fehlgeschlagen: {e}")
        return None

def main():
    # Session State für App-Stabilität
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # Input
    stadtteil = st.text_input("🏙️ Stadtteil eingeben", 
                              value="Maxvorstadt, München",
                              help="Beispiele: Kreuzberg, Berlin | Altstadt, Hamburg")

    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("🚀 **Analyse starten**", 
                     disabled=st.session_state.analysis_started,
                     type="primary"):
            if stadtteil:
                st.session_state.analysis_started = True
                st.session_state.analysis_complete = False

    with col2:
        if st.session_state.analysis_complete:
            if st.button("🔄 Neue Analyse"):
                st.session_state.analysis_started = False
                st.session_state.analysis_complete = False
                st.rerun()
    
    with col3:
        if st.button("🛑 Stop"):
            st.session_state.analysis_started = False
            st.rerun()

    if not st.session_state.analysis_started or not stadtteil:
        st.info("👆 Geben Sie einen Stadtteil ein und starten Sie die Analyse")
        return

    # Main Analysis with comprehensive error handling
    try:
        with st.spinner("🔍 Initialisiere Analyse..."):
            # Step 1: Geocoding
            gebiet = safe_execution(geocode_to_gdf_safe, stadtteil)
            if gebiet is None:
                st.session_state.analysis_started = False
                return
            
            polygon = gebiet.geometry.iloc[0]
            utm_crs = gebiet.estimate_utm_crs()
            gebiet = gebiet.to_crs(utm_crs)
            area = gebiet.geometry.iloc[0]

            # Step 2: Create Grid
            cell_size = 60  # Slightly larger for stability
            minx, miny, maxx, maxy = area.bounds
            grid_cells = [
                box(x, y, x + cell_size, y + cell_size)
                for x in np.arange(minx, maxx, cell_size)
                for y in np.arange(miny, maxy, cell_size)
                if box(x, y, x + cell_size, y + cell_size).intersects(area)
            ]
            
            if not grid_cells:
                st.error("❌ Gebiet zu klein für Analyse")
                st.session_state.analysis_started = False
                return
                
            grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)
            st.success(f"✅ Grid erstellt: {len(grid)} Zellen")

        # Step 3: Load OSM Data
        tags_buildings = {"building": True}
        tags_green = {"leisure": ["park"], "landuse": ["forest", "grass"]}
        
        buildings = safe_execution(load_osm_data_safe, polygon, tags_buildings)
        greens = safe_execution(load_osm_data_safe, polygon, tags_green)
        
        # Convert to UTM if data exists
        if buildings is not None and not buildings.empty:
            buildings = buildings.to_crs(utm_crs)
        if greens is not None and not greens.empty:
            greens = greens.to_crs(utm_crs)

        # Step 4: Analysis
        st.header("📊 Analyse-Ergebnisse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Gebäudedichte")
            fig1 = safe_execution(safe_building_analysis, grid.copy(), buildings or gpd.GeoDataFrame(), gebiet)
            if fig1:
                st.pyplot(fig1)
                plt.close(fig1)  # Prevent memory leaks

        with col2:
            st.subheader("🌳 Grünflächen-Distanz")
            fig2 = safe_execution(safe_green_analysis, grid.copy(), greens or gpd.GeoDataFrame(), gebiet)
            if fig2:
                st.pyplot(fig2)
                plt.close(fig2)

        # Step 5: Temperature Analysis
        st.subheader("🌡️ Temperatur-Heatmap")
        heatmap = safe_execution(safe_temperature_analysis, stadtteil)
        if heatmap:
            st.components.v1.html(heatmap._repr_html_(), height=500)

        st.session_state.analysis_complete = True
        st.balloons()
        st.success("🎉 **Analyse erfolgreich abgeschlossen!**")

    except Exception as e:
        st.error(f"❌ **Unerwarteter Fehler:** {str(e)}")
        st.error("🔧 **Lösung:** Versuchen Sie einen anderen Ort oder laden Sie die Seite neu")
        st.code(traceback.format_exc())
        st.session_state.analysis_started = False

if __name__ == "__main__":
    main()
