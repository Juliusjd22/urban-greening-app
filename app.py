import streamlit as st
import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from requests.adapters import HTTPAdapter, Retry
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
from sklearn.cluster import KMeans
import stackstac
import planetary_computer
from pystac_client import Client
from matplotlib.patches import Patch
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure OSMnx to be more resilient
ox.config(use_cache=True, log_console=False)

# Global session for efficient requests
session = requests.Session()
retries = Retry(
    total=5,  # Increased retries
    backoff_factor=2,  # Longer backoff
    status_forcelist=[429, 500, 502, 503, 504, 408],  # Added timeout error
    raise_on_status=False
)
session.mount('https://', HTTPAdapter(max_retries=retries))
session.mount('http://', HTTPAdapter(max_retries=retries))

def safe_geocode(location_name, max_retries=3):
    """
    Robust geocoding with multiple fallback methods
    """
    # Try multiple geocoding services
    geocoders = [
        Nominatim(user_agent="frigis-app-v2", timeout=15),
        # Add more geocoders as fallback if needed
    ]
    
    for i, geolocator in enumerate(geocoders):
        for retry in range(max_retries):
            try:
                st.info(f"🔍 Attempting geocoding (try {retry + 1}/{max_retries})...")
                location = geolocator.geocode(location_name, timeout=20)
                if location:
                    st.success(f"✅ Successfully geocoded: {location_name}")
                    return location
                else:
                    st.warning(f"⚠️ Location not found: {location_name}")
                    return None
            except Exception as e:
                st.warning(f"⚠️ Geocoding attempt {retry + 1} failed: {str(e)}")
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)  # Exponential backoff
                continue
    
    return None

def safe_osmnx_geocode(location_name, max_retries=3):
    """
    Robust OSMnx geocoding with retries
    """
    for retry in range(max_retries):
        try:
            st.info(f"🗺️ Loading area data (try {retry + 1}/{max_retries})...")
            
            # Set custom nominatim endpoint if needed
            ox.settings.nominatim_endpoint = "https://nominatim.openstreetmap.org"
            
            result = ox.geocode_to_gdf(location_name)
            st.success(f"✅ Successfully loaded area: {location_name}")
            return result
        except Exception as e:
            st.warning(f"⚠️ OSMnx geocoding attempt {retry + 1} failed: {str(e)}")
            if retry < max_retries - 1:
                time.sleep(3 ** retry)  # Exponential backoff
            continue
    
    return None

def safe_osmnx_features(polygon, tags, max_retries=3):
    """
    Robust OSMnx feature extraction with retries
    """
    for retry in range(max_retries):
        try:
            result = ox.features_from_polygon(polygon, tags=tags)
            return result
        except Exception as e:
            st.warning(f"⚠️ Feature extraction attempt {retry + 1} failed: {str(e)}")
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
            continue
    
    # Return empty GeoDataFrame if all retries fail
    return gpd.GeoDataFrame()

# Sidebar navigation
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
    try:
        from PIL import Image
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image("logo.png", width=60)
        with col2:
            st.markdown("<h1 style='margin-bottom: 0;'>friGIS</h1>", unsafe_allow_html=True)
    except:
        st.markdown("<h1 style='margin-bottom: 0;'>friGIS</h1>", unsafe_allow_html=True)
    
    def gebaeudedichte_analysieren_und_plotten(grid, buildings, gebiet):
        if buildings.empty:
            st.warning("⚠️ No building data available for this area.")
            # Create empty plot
            fig, ax = plt.subplots(figsize=(8, 8))
            gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
            ax.set_title("1️⃣ Building Density (No data available)")
            ax.axis("equal")
            plt.tight_layout()
            return fig
            
        progress = st.progress(0, text="🏗️ Calculating building density...")
        
        try:
            intersecting_geometries = buildings.sindex
            total = len(grid)
            
            for i, cell in enumerate(grid.geometry):
                try:
                    possible = list(intersecting_geometries.intersection(cell.bounds))
                    intersecting = buildings.iloc[possible][buildings.iloc[possible].intersects(cell)]
                    grid.at[i, "building_ratio"] = intersecting.intersection(cell).area.sum() / cell.area if not intersecting.empty else 0
                except:
                    grid.at[i, "building_ratio"] = 0
                    
                if i % max(1, total // 10) == 0:
                    progress.progress(i / total, text="🏗️ Calculating building density...")
                    
        except Exception as e:
            st.error(f"Error calculating building density: {e}")
            grid["building_ratio"] = 0
            
        progress.progress(1.0, text="🏗️ Building density calculated.")
        progress.empty()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        grid.plot(ax=ax, column="building_ratio", cmap="Reds", legend=True,
                  edgecolor="grey", linewidth=0.2)
        if not buildings.empty:
            buildings.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
        ax.set_title("1️⃣ Building Density (Red = dense)")
        ax.axis("equal")
        plt.tight_layout()
        return fig

    def distanz_zu_gruenflaechen_analysieren_und_plotten(grid, greens, gebiet, max_dist=500):
        progress = st.progress(0, text="🌳 Calculating distance to green areas...")
        
        if greens.empty:
            st.warning("⚠️ No green space data available for this area.")
            grid["dist_to_green"] = max_dist
            grid["score_distance_norm"] = 1.0
        else:
            try:
                greens_union = greens.geometry.union_all()
                total = len(grid)
                
                for i, geom in enumerate(grid.geometry):
                    try:
                        dist = greens_union.distance(geom.centroid)
                        grid.at[i, "dist_to_green"] = dist
                    except:
                        grid.at[i, "dist_to_green"] = max_dist
                        
                    if i % max(1, total // 10) == 0:
                        progress.progress(i / total, text="🌳 Calculating distance to green areas...")
                        
                grid["score_distance_norm"] = np.clip(grid["dist_to_green"] / max_dist, 0, 1)
            except Exception as e:
                st.error(f"Error calculating green distance: {e}")
                grid["dist_to_green"] = max_dist
                grid["score_distance_norm"] = 1.0
        
        progress.progress(1.0, text="🌳 Distance to green calculated.")
        progress.empty()
        
        cmap = plt.cm.Reds
        norm = mcolors.Normalize(vmin=0, vmax=1)
        fig, ax = plt.subplots(figsize=(8, 8))
        grid.plot(ax=ax, column="score_distance_norm", cmap=cmap, norm=norm,
                  edgecolor="grey", linewidth=0.2, legend=True,
                  legend_kwds={"label": "Distance to green (Red = far)"})
        if not greens.empty:
            greens.plot(ax=ax, color="green", alpha=0.5, edgecolor="darkgreen")
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
        ax.set_title("2️⃣ Distance to Green Areas")
        ax.axis("equal")
        plt.tight_layout()
        return fig

    def heatmap_mit_temperaturdifferenzen(ort_name, jahr=2022, radius_km=1.5, resolution_km=1.0):
        location = safe_geocode(ort_name)
        if not location:
            st.error("🌍 Geocoding failed - unable to get location coordinates")
            return None
    
        lat0, lon0 = location.latitude, location.longitude
        lats = np.arange(lat0 - radius_km / 111, lat0 + radius_km / 111 + 1e-6, resolution_km / 111)
        lons = np.arange(lon0 - radius_km / 85, lon0 + radius_km / 85 + 1e-6, resolution_km / 85)
    
        punkt_daten = []
        ref_temp = None
        total_points = len(lats) * len(lons)
        progress = st.progress(0, text="🔄 Loading temperature data...")
        count = 0
    
        for lat in lats:
            for lon in lons:
                success = False
                for retry in range(3):
                    try:
                        url = (
                            f"https://archive-api.open-meteo.com/v1/archive?"
                            f"latitude={lat}&longitude={lon}"
                            f"&start_date={jahr}-06-01&end_date={jahr}-08-31"
                            f"&daily=temperature_2m_max&timezone=auto"
                        )
                        r = session.get(url, timeout=15)
                        if r.status_code != 200:
                            time.sleep(2 ** retry)
                            continue
                        temps = r.json().get("daily", {}).get("temperature_2m_max", [])
                        if not temps:
                            break
                        avg_temp = round(np.mean(temps), 2)
                        punkt_daten.append([lat, lon, avg_temp])
    
                        if abs(lat - lat0) < resolution_km / 222 and abs(lon - lon0) < resolution_km / 170:
                            ref_temp = avg_temp
                        success = True
                        break
                    except Exception as e:
                        if retry < 2:
                            time.sleep(2 ** retry)
                        continue
                        
                count += 1
                progress.progress(min(count / total_points, 1.0), text="🔄 Loading temperature data...")
    
        progress.empty()
    
        if not punkt_daten or ref_temp is None:
            st.warning("⚠️ Not enough temperature data or center point value not available.")
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
    
    def analysiere_reflektivitaet_graustufen(stadtteil_name, n_clusters=5, year_range="2020-01-01/2024-12-31"):
        try:
            progress = st.progress(0, text="🔍 Searching for satellite data...")
            ort = safe_osmnx_geocode(stadtteil_name)
            if ort is None:
                st.error("Unable to geocode location for satellite analysis")
                return None
                
            bbox = ort.total_bounds
            progress.progress(0.1, text="🔍 Searching for Sentinel-2 data...")
        
            catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox.tolist(),
                datetime=year_range,
                query={"eo:cloud_cover": {"lt": 20}}
            )
            items = list(search.get_items())
            if not items:
                st.warning("❌ No suitable Sentinel-2 image found.")
                progress.empty()
                return None
        
            item = planetary_computer.sign(items[0])
            utm_crs = ort.estimate_utm_crs().to_epsg()
            progress.progress(0.4, text="🛰️ Loading image data...")
        
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
            progress.progress(0.7, text="🔢 Performing k-Means clustering...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
            labels = kmeans.labels_
        
            cluster_info = []
            for i in range(n_clusters):
                cluster_pixels = pixels[labels == i]
                if len(cluster_pixels) == 0:
                    cluster_info.append((i, 0, "No data"))
                    continue
                helligkeit = cluster_pixels.mean(axis=1).mean() / 255
                beschreibung = (
                    "🌞 Very bright (high reflectivity)" if helligkeit > 0.75 else
                    "🔆 Bright (moderately reflective)" if helligkeit > 0.5 else
                    "🌥️ Medium (neutral)" if helligkeit > 0.35 else
                    "🌡️ Dark (high heating potential)"
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
            st.error(f"Satellite analysis failed: {e}")
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

        # Session State initialization
        if 'analysis_started' not in st.session_state:
            st.session_state.analysis_started = False
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False

        stadtteil = st.text_input("🏙️ Enter neighborhood name", value="Maxvorstadt, München")

        # Button Logic with Session State
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Start Analysis", disabled=st.session_state.analysis_started):
                if stadtteil:
                    st.session_state.analysis_started = True
                    st.session_state.analysis_complete = False

        with col2:
            if st.session_state.analysis_complete:
                if st.button("🔄 New Analysis"):
                    st.session_state.analysis_started = False
                    st.session_state.analysis_complete = False
                    st.rerun()

        # Only run analysis when started
        if not st.session_state.analysis_started or not stadtteil:
            return

        # Show status
        if not st.session_state.analysis_complete:
            st.info("🔄 Analysis running...")

        # Load area data with error handling
        gebiet = safe_osmnx_geocode(stadtteil)
        if gebiet is None:
            st.error(f"📍 Area could not be loaded: {stadtteil}")
            st.error("Please check your internet connection and try again.")
            st.session_state.analysis_started = False
            return

        try:
            polygon = gebiet.geometry.iloc[0]
            utm_crs = gebiet.estimate_utm_crs()
            gebiet = gebiet.to_crs(utm_crs)
            area = gebiet.geometry.iloc[0].buffer(0)

            # Load building and green space data with error handling
            tags_buildings = {"building": True}
            tags_green = {
                "leisure": ["park", "garden"],
                "landuse": ["grass", "meadow", "forest"],
                "natural": ["wood", "tree_row", "scrub"]
            }
            
            st.info("📊 Loading building and green space data...")
            buildings = safe_osmnx_features(polygon, tags_buildings)
            greens = safe_osmnx_features(polygon, tags_green)
            
            # Convert to UTM and clean data
            if not buildings.empty:
                buildings = buildings.to_crs(utm_crs)
                buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty]
            
            if not greens.empty:
                greens = greens.to_crs(utm_crs)
                greens = greens[greens.geometry.is_valid & ~greens.geometry.is_empty]

            # Create analysis grid
            cell_size = 50
            minx, miny, maxx, maxy = area.bounds
            grid_cells = [
                box(x, y, x + cell_size, y + cell_size)
                for x in np.arange(minx, maxx, cell_size)
                for y in np.arange(miny, maxy, cell_size)
                if box(x, y, x + cell_size, y + cell_size).intersects(area)
            ]
            grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)

            # Run analyses
            st.subheader("Building Density")
            fig1 = gebaeudedichte_analysieren_und_plotten(grid, buildings, gebiet)
            st.pyplot(fig1)

            st.subheader("Distance to Green Areas")
            fig2 = distanz_zu_gruenflaechen_analysieren_und_plotten(grid, greens, gebiet)
            st.pyplot(fig2)

            st.subheader("Temperature Difference Heatmap")
            heatmap = heatmap_mit_temperaturdifferenzen(ort_name=stadtteil)
            if heatmap:
                st.components.v1.html(heatmap._repr_html_(), height=600)
            else:
                st.warning("No temperature data found.")

            st.subheader("k-Means Cluster Analysis of Satellite Data")
            fig3 = analysiere_reflektivitaet_graustufen(stadtteil, n_clusters=5)
            if fig3:
                st.pyplot(fig3)

            # Mark analysis as complete
            st.session_state.analysis_complete = True
            st.success("✅ Analysis completed! You can now start a new analysis.")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.session_state.analysis_started = False
            return
    
    # Call main function when on the main app page
    main()
