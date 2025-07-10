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
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# OpenCageData API Key
OPENCAGE_API_KEY = "bb1eb77da8504268a285bc3a82daa835"

def geocode_to_gdf_with_fallback(location_name):
    """Geocodierung mit OpenCageData, Fallback auf OSMnx"""
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
                offset = 0.01
                minx, miny, maxx, maxy = lon - offset, lat - offset, lon + offset, lat + offset
            
            polygon = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            gdf = gpd.GeoDataFrame(
                {'geometry': [polygon], 'name': [location_name]}, 
                crs='EPSG:4326'
            )
            st.info("✅ OpenCageData verwendet")
            return gdf
    except Exception as e:
        st.warning(f"⚠️ OpenCageData failed: {e}")
    
    try:
        st.info("🔄 Fallback auf OSMnx...")
        gdf = ox.geocode_to_gdf(location_name)
        st.info("✅ OSMnx Fallback erfolgreich")
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
    
    def load_osm_data_with_retry(polygon, tags, max_retries=3):
        """OSM Daten mit Retry-Logik laden - ORIGINAL QUALITÄT"""
        for attempt in range(max_retries):
            try:
                data = ox.features_from_polygon(polygon, tags=tags)
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"OSM Versuch {attempt + 1} fehlgeschlagen, versuche erneut...")
                    time.sleep(2 ** attempt)
                else:
                    st.error(f"OSM Daten konnten nach {max_retries} Versuchen nicht geladen werden: {e}")
                    return gpd.GeoDataFrame()
    
    def gebaeudedichte_analysieren_und_plotten(grid, buildings, gebiet):
        """ORIGINAL Gebäudedichte-Funktion mit kleinen Performance-Verbesserungen"""
        if buildings.empty:
            st.warning("⚠️ Keine Gebäudedaten verfügbar")
            grid["building_ratio"] = 0.0
        else:
            progress = st.progress(0, text="🏗️ Calculating building density...")
            intersecting_geometries = buildings.sindex
            total = len(grid)
            
            # Batch-Processing für bessere Performance
            batch_size = max(1, total // 20)  # 20 Updates statt 10
            
            for i, cell in enumerate(grid.geometry):
                try:
                    possible = list(intersecting_geometries.intersection(cell.bounds))
                    intersecting = buildings.iloc[possible][buildings.iloc[possible].intersects(cell)]
                    grid.at[i, "building_ratio"] = intersecting.intersection(cell).area.sum() / cell.area if not intersecting.empty else 0
                except Exception:
                    grid.at[i, "building_ratio"] = 0
                    
                if i % batch_size == 0:
                    progress.progress(i / total, text="🏗️ Calculating building density...")
            
            progress.progress(1.0, text="🏗️ Building density calculated.")
            progress.empty()
        
        # Original Plot-Qualität beibehalten
        fig, ax = plt.subplots(figsize=(8, 8))
        grid.plot(ax=ax, column="building_ratio", cmap="Reds", legend=True,
                  edgecolor="grey", linewidth=0.2)
        buildings.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
        ax.set_title("1️⃣ Building Density (Red = dense)")
        ax.axis("equal")
        plt.tight_layout()
        return fig

    def distanz_zu_gruenflaechen_analysieren_und_plotten(grid, greens, gebiet, max_dist=500):
        """ORIGINAL Grünflächen-Funktion mit kleinen Performance-Verbesserungen"""
        if greens.empty:
            st.warning("⚠️ Keine Grünflächendaten verfügbar")
            grid["dist_to_green"] = max_dist
            grid["score_distance_norm"] = 1.0
        else:
            progress = st.progress(0, text="🌳 Calculating distance to green areas...")
            greens_union = greens.geometry.union_all()
            total = len(grid)
            
            batch_size = max(1, total // 20)
            
            for i, geom in enumerate(grid.geometry):
                try:
                    dist = greens_union.distance(geom.centroid)
                    grid.at[i, "dist_to_green"] = dist
                except Exception:
                    grid.at[i, "dist_to_green"] = max_dist
                    
                if i % batch_size == 0:
                    progress.progress(i / total, text="🌳 Calculating distance to green areas...")
            
            grid["score_distance_norm"] = np.clip(grid["dist_to_green"] / max_dist, 0, 1)
            progress.progress(1.0, text="🌳 Distance to green calculated.")
            progress.empty()
        
        # Original Plot-Qualität
        cmap = plt.cm.Reds
        norm = mcolors.Normalize(vmin=0, vmax=1)
        fig, ax = plt.subplots(figsize=(8, 8))
        grid.plot(ax=ax, column="score_distance_norm", cmap=cmap, norm=norm,
                  edgecolor="grey", linewidth=0.2, legend=True,
                  legend_kwds={"label": "Distance to green (Red = far)"})
        greens.plot(ax=ax, color="green", alpha=0.5, edgecolor="darkgreen")
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
        ax.set_title("2️⃣ Distance to Green Areas")
        ax.axis("equal")
        plt.tight_layout()
        return fig

    def heatmap_mit_temperaturdifferenzen_erweitert(ort_name, jahr=2022, radius_km=1.2, resolution_km=1.0):
        """ERWEITERTE Temperaturdaten - MEHR Punkte als Original"""
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
        
        # ERWEITERTE Parameter für mehr Temperaturpunkte
        lats = np.arange(lat0 - radius_km / 111, lat0 + radius_km / 111 + 1e-6, resolution_km / 111)
        lons = np.arange(lon0 - radius_km / 85, lon0 + radius_km / 85 + 1e-6, resolution_km / 85)
    
        punkt_daten = []
        ref_temp = None
        total_points = len(lats) * len(lons)
        progress = st.progress(0, text=f"🔄 Lade {total_points} Temperaturpunkte (erweitert)...")
        count = 0
        
        def fetch_temperature(lat, lon):
            for _ in range(2):
                try:
                    url = (
                        f"https://archive-api.open-meteo.com/v1/archive?"
                        f"latitude={lat}&longitude={lon}"
                        f"&start_date={jahr}-06-01&end_date={jahr}-08-31"
                        f"&daily=temperature_2m_max&timezone=auto"
                    )
                    r = session.get(url, timeout=8)
                    if r.status_code != 200:
                        time.sleep(0.5)
                        continue
                    temps = r.json().get("daily", {}).get("temperature_2m_max", [])
                    if not temps:
                        break
                    return lat, lon, round(np.mean(temps), 2)
                except Exception:
                    time.sleep(0.5)
            return lat, lon, None
        
        # Erhöhte Parallelität für mehr Datenpunkte
        coords = [(lat, lon) for lat in lats for lon in lons]
        with ThreadPoolExecutor(max_workers=6) as executor:  # Erhöht von 5 auf 6
            futures = [executor.submit(fetch_temperature, lat, lon) for lat, lon in coords]
            
            for future in as_completed(futures):
                lat, lon, temp = future.result()
                if temp is not None:
                    punkt_daten.append([lat, lon, temp])
                    
                    if abs(lat - lat0) < resolution_km / 222 and abs(lon - lon0) < resolution_km / 170:
                        ref_temp = temp
                
                count += 1
                if count % 3 == 0:  # Häufigere Updates für besseres Feedback
                    progress.progress(min(count / total_points, 1.0), 
                                   text=f"🔄 Temperaturdaten: {count}/{total_points}")
    
        progress.empty()
    
        if not punkt_daten:
            st.warning("⚠️ Keine Temperaturdaten verfügbar.")
            return None
            
        if ref_temp is None:
            ref_temp = np.mean([temp for _, _, temp in punkt_daten])
            st.info("ℹ️ Referenztemperatur geschätzt")
    
        differenzpunkte = [
            [lat, lon, round(temp - ref_temp, 2)]
            for lat, lon, temp in punkt_daten
        ]
    
        # Verbesserte Heatmap mit mehr Datenpunkten
        m = folium.Map(location=[lat0, lon0], zoom_start=13, tiles="CartoDB positron")
        HeatMap(
            [[lat, lon, abs(diff)] for lat, lon, diff in differenzpunkte],
            radius=20,  # Etwas größer für bessere Sichtbarkeit
            blur=25,
            max_zoom=13,
            gradient={0.0: "green", 0.3: "lightyellow", 0.6: "orange", 1.0: "red"}
        ).add_to(m)
    
        # Mehr Temperaturmarker
        for lat, lon, diff in differenzpunkte:
            sign = "+" if diff > 0 else ("−" if diff < 0 else "±")
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(html=f"<div style='font-size:10pt; color:black'><b>{sign}{abs(diff):.2f}°C</b></div>")
            ).add_to(m)
    
        st.success(f"✅ {len(punkt_daten)} Temperaturpunkte geladen (erweitert)!")
        return m
    
    def analysiere_reflektivitaet_graustufen(stadtteil_name, n_clusters=5, year_range="2020-01-01/2024-12-31"):
        """Original Satellitendatenanalyse mit kleinen Performance-Verbesserungen"""
        try:
            progress = st.progress(0, text="🔍 Satellitendaten werden gesucht...")
            
            gebiet = geocode_to_gdf_with_fallback(stadtteil_name)
            if gebiet is None:
                st.warning("❌ Gebiet konnte nicht gefunden werden.")
                progress.empty()
                return None
                
            bbox = gebiet.total_bounds
            progress.progress(0.1, text="🔍 Suche nach Sentinel-2 Daten...")
        
            catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox.tolist(),
                datetime=year_range,
                query={"eo:cloud_cover": {"lt": 20}}
            )
            items = list(search.get_items())
            if not items:
                st.warning("❌ Kein geeignetes Sentinel-2 Bild gefunden.")
                progress.empty()
                return None
        
            item = planetary_computer.sign(items[0])
            utm_crs = gebiet.estimate_utm_crs().to_epsg()
            progress.progress(0.4, text="🛰️ Bilddaten werden geladen...")
        
            # Leicht reduzierte Auflösung für bessere Performance, aber immer noch gute Qualität
            stack = stackstac.stack(
                [item],
                assets=["B04", "B03", "B02"],
                resolution=15,  # Reduziert von 10 auf 15
                bounds_latlon=bbox.tolist(),
                epsg=utm_crs
            )
            rgb = stack.isel(band=[0,1,2], time=0).transpose("y","x","band").values
            rgb = np.nan_to_num(rgb)
            rgb_scaled = np.clip((rgb / 3000) * 255, 0, 255).astype(np.uint8)
        
            h, w, _ = rgb_scaled.shape
            pixels = rgb_scaled.reshape(-1, 3)
            progress.progress(0.7, text="🔢 k-Means Clustering wird durchgeführt...")
            
            # Etwas optimiertes k-Means, aber Original-Qualität
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=8).fit(pixels)  # Reduziert von 10 auf 8
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
            progress.empty()
            return fig
        except Exception as e:
            st.error(f"Satellitendatenanalyse fehlgeschlagen: {e}")
            return None
    
    def main():
        st.markdown("""
            by Philippa, Samuel, Julius  
            Take a look at our interactive prototype designed to demonstrate 
            how environmental and geospatial data can be used to identify 
            urban areas in need of greening interventions. It integrates 
            multiple open-source datasets and satellite sources to analyze 
            urban heat and greening potential at the neighborhood level.
        """)

        # Session State initialisieren
        if 'analysis_started' not in st.session_state:
            st.session_state.analysis_started = False
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False

        stadtteil = st.text_input("🏙️ Stadtteilname eingeben", value="Maxvorstadt, München")

        # Button Logic mit Session State
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Analyse starten", disabled=st.session_state.analysis_started):
                if stadtteil:
                    st.session_state.analysis_started = True
                    st.session_state.analysis_complete = False

        with col2:
            if st.session_state.analysis_complete:
                if st.button("🔄 Neue Analyse"):
                    st.session_state.analysis_started = False
                    st.session_state.analysis_complete = False
                    st.rerun()

        # Analyse nur ausführen wenn gestartet
        if not st.session_state.analysis_started or not stadtteil:
            return

        # Status anzeigen
        if not st.session_state.analysis_complete:
            st.info("🔄 Analyse läuft...")

        try:
            gebiet = geocode_to_gdf_with_fallback(stadtteil)
            if gebiet is None:
                st.error("📍 Gebiet konnte nicht gefunden werden.")
                st.session_state.analysis_started = False
                return
                
        except Exception as e:
            st.error(f"📍 Unerwarteter Fehler: {e}")
            st.session_state.analysis_started = False
            return

        polygon = gebiet.geometry.iloc[0]
        utm_crs = gebiet.estimate_utm_crs()
        gebiet = gebiet.to_crs(utm_crs)
        area = gebiet.geometry.iloc[0].buffer(0)

        # Original OSM-Tags für vollständige Daten
        tags_buildings = {"building": True}
        tags_green = {
            "leisure": ["park", "garden"],
            "landuse": ["grass", "meadow", "forest"],
            "natural": ["wood", "tree_row", "scrub"]
        }
        
        st.info("📡 Lade OSM-Daten...")
        buildings = load_osm_data_with_retry(polygon, tags_buildings)
        greens = load_osm_data_with_retry(polygon, tags_green)
        
        # Daten bereinigen
        if not buildings.empty:
            buildings = buildings.to_crs(utm_crs)
            buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty]
        if not greens.empty:
            greens = greens.to_crs(utm_crs)
            greens = greens[greens.geometry.is_valid & ~greens.geometry.is_empty]

        # Original Grid-Größe für Qualität
        cell_size = 50
        minx, miny, maxx, maxy = area.bounds
        grid_cells = [
            box(x, y, x + cell_size, y + cell_size)
            for x in np.arange(minx, maxx, cell_size)
            for y in np.arange(miny, maxy, cell_size)
            if box(x, y, x + cell_size, y + cell_size).intersects(area)
        ]
        grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)

        # Analysen durchführen
        st.subheader("Gebäudedichte")
        fig1 = gebaeudedichte_analysieren_und_plotten(grid.copy(), buildings, gebiet)
        st.pyplot(fig1)

        st.subheader("Distanz zu Grünflächen")
        fig2 = distanz_zu_gruenflaechen_analysieren_und_plotten(grid.copy(), greens, gebiet)
        st.pyplot(fig2)

        st.subheader("Temperaturdifferenz Heatmap (Erweitert)")
        heatmap = heatmap_mit_temperaturdifferenzen_erweitert(ort_name=stadtteil)
        if heatmap:
            st.components.v1.html(heatmap._repr_html_(), height=600)
        else:
            st.warning("Keine Temperaturdaten gefunden.")

        st.subheader("k-Means Clusteranalyse von Satellitendaten")
        fig3 = analysiere_reflektivitaet_graustufen(stadtteil, n_clusters=5)
        if fig3:
            st.pyplot(fig3)

        # Am Ende der Analyse
        st.session_state.analysis_complete = True
        st.success("✅ Analyse abgeschlossen! Du kannst jetzt eine neue Analyse starten.")
    
    main()
