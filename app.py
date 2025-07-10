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
warnings.filterwarnings("ignore", category=UserWarning)

# Globale Session für effiziente Requests
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# OpenCageData API Key
OPENCAGE_API_KEY = "bb1eb77da8504268a285bc3a82daa835"

def geocode_to_gdf_with_fallback(location_name):
    """Schnelle Geocodierung ohne Cache"""
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
                offset = 0.005  # Kleineres Gebiet für bessere Performance
                minx, miny, maxx, maxy = lon - offset, lat - offset, lon + offset, lat + offset
            
            polygon = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            gdf = gpd.GeoDataFrame(
                {'geometry': [polygon], 'name': [location_name]}, 
                crs='EPSG:4326'
            )
            return gdf
    except Exception as e:
        st.warning(f"⚠️ OpenCageData failed: {e}")
    
    # Fallback auf OSMnx
    try:
        gdf = ox.geocode_to_gdf(location_name)
        return gdf
    except Exception as e:
        st.error(f"❌ Beide Geocoding-Services fehlgeschlagen: {e}")
        return None

# Seitenleiste mit Navigation
page = st.sidebar.radio("🔍 Select Analysis or Info Page", [
    "🏠 Main App",
    "🏗️ Building Density – Info",
    "🌳 Distance to Green – Info", 
    "🔥 Heatmap – Info",
    "🛰️ Satellite k-Means – Info",
    "🌱 What We Plan Next",
    "🐞 Report a Bug"
])

if page == "🏗️ Building Density – Info":
    st.title("🏗️ Building Density – Info")
    st.markdown("""
    **How it works:**
    Building footprints from OpenStreetMap are used to calculate the ratio of built area per cell.
    
    **Why it's useful:**
    High building density often correlates with heat accumulation in cities. This metric helps identify particularly heat-stressed urban zones.
    """)

elif page == "🌳 Distance to Green – Info":
    st.title("🌳 Distance to Green – Info")
    st.markdown("""
    **How it works:**
    We calculate the distance from each urban grid cell to the nearest green space.

    **Why it's important:**
    Proximity to green areas directly influences local cooling and microclimates. Areas far from green are heat-prone.
    """)

elif page == "🔥 Heatmap – Info":
    st.title("🔥 Heatmap – Info")
    st.markdown("""
    **How it works:**
    Daily maximum temperatures (from Open-Meteo) are collected in a grid. Differences from the central point show relative heating.

    **Why it's valuable:**
    It helps identify local hotspots and temperature variations within neighborhoods.
    """)

elif page == "🛰️ Satellite k-Means – Info":
    st.title("🛰️ Satellite k-Means – Info")
    st.markdown("""
    **How it works:**
    Satellite imagery (Sentinel-2) is clustered by brightness to assess reflectivity and infer potential for surface heating.

    **Note:**
    This only works in large urban areas.
    """)

elif page == "🌱 What We Plan Next":
    st.title("🌱 What's Next")
    st.markdown("""
    **Tailored Greening Plans:**
    Based on the data, we aim to generate custom recommendations for tree planting, rooftop/vertical greening, etc. per location.

    **Monitoring:**
    We plan to integrate sensors or satellite data to track cooling impacts over time.
    """)

elif page == "🐞 Report a Bug":
    st.title("🐞 Report a Bug or Issue")
    st.markdown("""
    We've had some server issues in recent days.
    
    👉 If something doesn't work or crashes, please send a short message to:
    **julius.dickmann@muenchen.enactus.team**
    
    Thank you!
    """)

elif page == "🏠 Main App":
    from PIL import Image
    col1, col2 = st.columns([1, 6])
    with col1:
        try:
            st.image("logo.png", width=60)
        except:
            pass
    with col2:
        st.markdown("<h1 style='margin-bottom: 0;'>friGIS</h1>", unsafe_allow_html=True)
    
    def load_osm_data_fast(polygon, tags, max_retries=2):
        """Schnelles OSM Daten laden ohne Cache"""
        for attempt in range(max_retries):
            try:
                # Reduzierte Timeout-Werte für schnellere Fehlerbehandlung
                data = ox.features_from_polygon(polygon, tags=tags, timeout=15)
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Reduzierte Wartezeit
                else:
                    st.warning(f"⚠️ OSM Daten nicht verfügbar: {str(e)[:50]}...")
                    return gpd.GeoDataFrame()
    
    def gebaeudedichte_analysieren_schnell(grid, buildings, gebiet):
        """Optimierte Gebäudedichte-Berechnung"""
        if buildings.empty:
            grid["building_ratio"] = 0.1
        else:
            # Vereinfachte Berechnung ohne Progress Bar für bessere Performance
            try:
                # Spatial Index für schnellere Suche
                spatial_index = buildings.sindex
                for i, cell in enumerate(grid.geometry):
                    possible_matches_index = list(spatial_index.intersection(cell.bounds))
                    if possible_matches_index:
                        possible_matches = buildings.iloc[possible_matches_index]
                        precise_matches = possible_matches[possible_matches.intersects(cell)]
                        if not precise_matches.empty:
                            intersection_area = precise_matches.intersection(cell).area.sum()
                            grid.at[i, "building_ratio"] = intersection_area / cell.area
                        else:
                            grid.at[i, "building_ratio"] = 0
                    else:
                        grid.at[i, "building_ratio"] = 0
            except:
                grid["building_ratio"] = 0.1
        
        # Schnelleres Plotting
        fig, ax = plt.subplots(figsize=(6, 6))  # Kleinere Figur
        grid.plot(ax=ax, column="building_ratio", cmap="Reds", legend=True,
                  edgecolor="none", linewidth=0)  # Keine Edges für Performance
        if not buildings.empty and len(buildings) < 1000:  # Nur bei wenigen Gebäuden anzeigen
            buildings.plot(ax=ax, color="lightgrey", alpha=0.3)
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=1)
        ax.set_title("1️⃣ Building Density", fontsize=12)
        ax.axis("off")  # Achsen ausblenden für sauberes Layout
        plt.tight_layout()
        return fig

    def distanz_zu_gruenflaechen_schnell(grid, greens, gebiet, max_dist=300):
        """Optimierte Grünflächen-Distanz"""
        if greens.empty:
            grid["dist_to_green"] = max_dist
            grid["score_distance_norm"] = 1.0
        else:
            try:
                # Vereinfachte Union-Berechnung
                if len(greens) > 50:  # Bei vielen Grünflächen: Sampling
                    greens_sample = greens.sample(n=50)
                    greens_union = greens_sample.geometry.union_all()
                else:
                    greens_union = greens.geometry.union_all()
                
                # Schnellere Distanzberechnung
                for i, geom in enumerate(grid.geometry):
                    try:
                        dist = greens_union.distance(geom.centroid)
                        grid.at[i, "dist_to_green"] = min(dist, max_dist)
                    except:
                        grid.at[i, "dist_to_green"] = max_dist
                
                grid["score_distance_norm"] = grid["dist_to_green"] / max_dist
            except:
                grid["dist_to_green"] = max_dist
                grid["score_distance_norm"] = 1.0
        
        fig, ax = plt.subplots(figsize=(6, 6))
        grid.plot(ax=ax, column="score_distance_norm", cmap="Reds",
                  edgecolor="none", linewidth=0, legend=True)
        if not greens.empty and len(greens) < 200:
            greens.plot(ax=ax, color="green", alpha=0.4)
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=1)
        ax.set_title("2️⃣ Distance to Green", fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def heatmap_temperaturen_erweitert(ort_name, jahr=2022, radius_km=1.2, resolution_km=0.8):
        """ERWEITERTE Temperaturdaten mit MEHR Punkten für bessere Heatmap"""
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        try:
            results = geocoder.geocode(ort_name, no_annotations=1)
        except Exception as e:
            st.error(f"🌍 Geokodierung fehlgeschlagen: {e}")
            return None

        if not results:
            st.warning("❗ Ort konnte nicht gefunden werden.")
            return None
    
        lat0, lon0 = results[0]['geometry']['lat'], results[0]['geometry']['lng']
        
        # MEHR Temperaturpunkte für detailliertere Heatmap
        lats = np.arange(lat0 - radius_km / 111, lat0 + radius_km / 111 + 1e-6, resolution_km / 111)
        lons = np.arange(lon0 - radius_km / 85, lon0 + radius_km / 85 + 1e-6, resolution_km / 85)
    
        punkt_daten = []
        ref_temp = None
        total_points = len(lats) * len(lons)
        
        progress = st.progress(0, text=f"🌡️ Lade {total_points} Temperaturpunkte...")
        count = 0
        
        def fetch_temperature_fast(lat, lon):
            """Optimierte Temperaturabfrage"""
            try:
                url = (
                    f"https://archive-api.open-meteo.com/v1/archive?"
                    f"latitude={lat}&longitude={lon}"
                    f"&start_date={jahr}-07-01&end_date={jahr}-07-31"  # Nur Juli für Performance
                    f"&daily=temperature_2m_max&timezone=auto"
                )
                r = session.get(url, timeout=6)
                if r.status_code == 200:
                    temps = r.json().get("daily", {}).get("temperature_2m_max", [])
                    if temps:
                        return lat, lon, round(np.mean(temps), 2)
            except:
                pass
            return lat, lon, None
        
        # ERHÖHTE Parallelität für mehr Temperaturpunkte
        coords = [(lat, lon) for lat in lats for lon in lons]
        with ThreadPoolExecutor(max_workers=8) as executor:  # Mehr Workers
            futures = [executor.submit(fetch_temperature_fast, lat, lon) for lat, lon in coords]
            
            for future in as_completed(futures):
                lat, lon, temp = future.result()
                if temp is not None:
                    punkt_daten.append([lat, lon, temp])
                    
                    # Referenztemperatur am Zentrum
                    if abs(lat - lat0) < resolution_km / 222 and abs(lon - lon0) < resolution_km / 170:
                        ref_temp = temp
                
                count += 1
                if count % 5 == 0:  # Weniger häufige Updates
                    progress.progress(min(count / total_points, 1.0), 
                                   text=f"🌡️ {count}/{total_points} Temperaturpunkte")
    
        progress.empty()
    
        if not punkt_daten:
            st.warning("⚠️ Keine Temperaturdaten verfügbar.")
            return None
            
        if ref_temp is None:
            ref_temp = np.mean([temp for _, _, temp in punkt_daten])
    
        # Temperaturdifferenzen berechnen
        differenzpunkte = [
            [lat, lon, round(temp - ref_temp, 2)]
            for lat, lon, temp in punkt_daten
        ]
    
        # Verbesserte Heatmap mit mehr Datenpunkten
        m = folium.Map(location=[lat0, lon0], zoom_start=14, tiles="CartoDB positron")
        
        # Intensivere Heatmap-Darstellung
        HeatMap(
            [[lat, lon, abs(diff) + 0.1] for lat, lon, diff in differenzpunkte],  # +0.1 für bessere Sichtbarkeit
            radius=20,  # Größerer Radius
            blur=15,    # Weniger Blur für schärfere Darstellung
            max_zoom=15,
            gradient={0.0: "blue", 0.2: "green", 0.5: "yellow", 0.8: "orange", 1.0: "red"}
        ).add_to(m)
    
        # Mehr Temperatur-Marker für bessere Information
        for lat, lon, diff in differenzpunkte[::2]:  # Jeden 2. Punkt anzeigen
            color = "red" if diff > 0.5 else "blue" if diff < -0.5 else "orange"
            sign = "+" if diff > 0 else ("−" if diff < 0 else "±")
            folium.CircleMarker(
                [lat, lon],
                radius=6,
                popup=f"{sign}{abs(diff):.1f}°C",
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
    
        st.success(f"✅ {len(punkt_daten)} Temperaturpunkte geladen!")
        return m
    
    def satellitendaten_schnell(stadtteil_name, n_clusters=4):
        """Optimierte Satellitendatenanalyse"""
        try:
            gebiet = geocode_to_gdf_with_fallback(stadtteil_name)
            if gebiet is None:
                return None
                
            bbox = gebiet.total_bounds
            
            # Schnellere STAC-Suche
            catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox.tolist(),
                datetime="2023-01-01/2024-12-31",  # Größerer Zeitraum
                query={"eo:cloud_cover": {"lt": 40}},  # Mehr Bilder zulassen
                limit=1  # Nur das erste Bild
            )
            items = list(search.get_items())
            if not items:
                st.warning("❌ Kein Satellitenbild gefunden.")
                return None
        
            item = planetary_computer.sign(items[0])
            utm_crs = gebiet.estimate_utm_crs().to_epsg()
        
            # Niedrigere Auflösung für bessere Performance
            stack = stackstac.stack(
                [item],
                assets=["B04", "B03", "B02"],
                resolution=60,  # Deutlich reduziert
                bounds_latlon=bbox.tolist(),
                epsg=utm_crs
            )
            rgb = stack.isel(band=[0,1,2], time=0).transpose("y","x","band").values
            rgb = np.nan_to_num(rgb)
            rgb_scaled = np.clip((rgb / 3000) * 255, 0, 255).astype(np.uint8)
        
            h, w, _ = rgb_scaled.shape
            pixels = rgb_scaled.reshape(-1, 3)
            
            # Schnelleres k-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100).fit(pixels)
            labels = kmeans.labels_
        
            # Cluster-Info berechnen
            cluster_info = []
            for i in range(n_clusters):
                cluster_pixels = pixels[labels == i]
                if len(cluster_pixels) == 0:
                    continue
                helligkeit = cluster_pixels.mean(axis=1).mean() / 255
                beschreibung = (
                    "🌞 Sehr hell" if helligkeit > 0.7 else
                    "🔆 Hell" if helligkeit > 0.5 else
                    "🌥️ Mittel" if helligkeit > 0.35 else
                    "🌡️ Dunkel"
                )
                cluster_info.append((i, round(helligkeit, 2), beschreibung))
        
            # Visualisierung
            gray_values = np.linspace(0, 255, n_clusters).astype(int)
            gray_colors = np.stack([gray_values]*3, axis=1)
            cluster_image = gray_colors[labels].reshape(h, w, 3).astype(np.uint8)
        
            fig, ax = plt.subplots(figsize=(5,5))  # Kleinere Figur
            ax.imshow(cluster_image)
            ax.axis("off")
        
            # Kompakte Legende
            legend_elements = [
                Patch(facecolor=gray_colors[i]/255, edgecolor='black',
                      label=f"{cluster_info[i][2]}")
                for i in range(len(cluster_info))
            ]
            ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.1),
                      ncol=2, frameon=True, fontsize="x-small")
            plt.tight_layout()
            return fig
        except Exception as e:
            st.warning(f"Satellitendaten nicht verfügbar: {str(e)[:50]}...")
            return None
    
    def main():
        st.markdown("""
            **🚀 Optimierte Version** - by Philippa, Samuel, Julius  
            Schnelle Analyse von städtischen Wärmeinseln und Begrünungspotentialen mit erweiterten Temperaturdaten.
        """)

        # Performance-Indikator
        start_time = time.time()

        # Session State
        if 'analysis_started' not in st.session_state:
            st.session_state.analysis_started = False
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False

        stadtteil = st.text_input("🏙️ Stadtteilname eingeben", value="Maxvorstadt, München")

        # Buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Schnellanalyse starten", disabled=st.session_state.analysis_started):
                if stadtteil:
                    st.session_state.analysis_started = True
                    st.session_state.analysis_complete = False

        with col2:
            if st.session_state.analysis_complete:
                if st.button("🔄 Neue Analyse"):
                    st.session_state.analysis_started = False
                    st.session_state.analysis_complete = False
                    st.rerun()

        if not st.session_state.analysis_started or not stadtteil:
            return

        st.info("⚡ Schnellanalyse läuft...")

        try:
            # Geocoding
            gebiet = geocode_to_gdf_with_fallback(stadtteil)
            if gebiet is None:
                st.error("📍 Gebiet konnte nicht gefunden werden.")
                st.session_state.analysis_started = False
                return
                
            polygon = gebiet.geometry.iloc[0]
            utm_crs = gebiet.estimate_utm_crs()
            gebiet = gebiet.to_crs(utm_crs)
            area = gebiet.geometry.iloc[0].buffer(0)

            # Optimierte OSM-Abfragen
            tags_buildings = {"building": True}
            tags_green = {
                "leisure": ["park", "garden"],
                "landuse": ["grass", "forest"],  # Reduziert
                "natural": ["wood"]  # Reduziert
            }
            
            # Parallele OSM-Abfragen für bessere Performance
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_buildings = executor.submit(load_osm_data_fast, polygon, tags_buildings)
                future_greens = executor.submit(load_osm_data_fast, polygon, tags_green)
                
                buildings = future_buildings.result()
                greens = future_greens.result()
            
            # Daten bereinigen
            if not buildings.empty:
                buildings = buildings.to_crs(utm_crs)
                buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty]
            if not greens.empty:
                greens = greens.to_crs(utm_crs)
                greens = greens[greens.geometry.is_valid & ~greens.geometry.is_empty]

            # Optimiertes Grid (größere Zellen für bessere Performance)
            cell_size = 75  # Vergrößert von 50
            minx, miny, maxx, maxy = area.bounds
            grid_cells = [
                box(x, y, x + cell_size, y + cell_size)
                for x in np.arange(minx, maxx, cell_size)
                for y in np.arange(miny, maxy, cell_size)
                if box(x, y, x + cell_size, y + cell_size).intersects(area)
            ]
            grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)

            # Analysen parallel durchführen
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏗️ Gebäudedichte")
                fig1 = gebaeudedichte_analysieren_schnell(grid.copy(), buildings, gebiet)
                st.pyplot(fig1, use_container_width=True)

            with col2:
                st.subheader("🌳 Grünflächen-Distanz")
                fig2 = distanz_zu_gruenflaechen_schnell(grid.copy(), greens, gebiet)
                st.pyplot(fig2, use_container_width=True)

            # Erweiterte Temperaturdaten (wichtigster Teil!)
            st.subheader("🔥 Erweiterte Temperatur-Heatmap")
            st.info("🌡️ Lade MEHR Temperaturpunkte für detailliertere Analyse...")
            heatmap = heatmap_temperaturen_erweitert(ort_name=stadtteil)
            if heatmap:
                st.components.v1.html(heatmap._repr_html_(), height=500)

            # Satellitendaten optional
            with st.expander("🛰️ Satellitendaten-Analyse (optional)"):
                if st.button("Satellitendaten laden"):
                    fig3 = satellitendaten_schnell(stadtteil)
                    if fig3:
                        st.pyplot(fig3, use_container_width=True)

            # Performance-Messung
            end_time = time.time()
            duration = round(end_time - start_time, 1)
            
            st.session_state.analysis_complete = True
            st.success(f"✅ Schnellanalyse abgeschlossen in {duration}s! 🚀")

        except Exception as e:
            st.error(f"❌ Fehler: {e}")
            st.session_state.analysis_started = False
    
    main()
