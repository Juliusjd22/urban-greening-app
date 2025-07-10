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
    """Geocodierung mit OpenCageData, Fallback auf OSMnx wenn nötig"""
    # Versuch 1: OpenCageData
    try:
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        results = geocoder.geocode(location_name, no_annotations=1)
        if results:
            result = results[0]
            
            if 'bounds' in result:
                bounds = result['bounds']
                minx, miny = bounds['southwest']['lng'], bounds['southwest']['lat']
                maxx, maxy = bounds['northeast']['lng'], bounds['northeast']['lat']
                # GRÖSSERER Radius für erste zwei Analysen
                center_lon = (minx + maxx) / 2
                center_lat = (miny + maxy) / 2
                offset = 0.008  # Erhöht von 0.006 auf 0.008 = ca. 800m Radius
                minx, miny, maxx, maxy = center_lon - offset, center_lat - offset, center_lon + offset, center_lat + offset
            else:
                lat, lon = result['geometry']['lat'], result['geometry']['lng']
                offset = 0.008  # Erhöht von 0.006 auf 0.008 = ca. 800m Radius
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
    
    # Versuch 2: OSMnx Fallback
    try:
        st.info("🔄 Fallback auf OSMnx...")
        gdf = ox.geocode_to_gdf(location_name)
        # Auch hier kleineres Gebiet für erste zwei Analysen
        bounds = gdf.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        offset = 0.008  # Gleicher Radius wie bei OpenCageData
        polygon = Polygon([(center_lon - offset, center_lat - offset), 
                         (center_lon + offset, center_lat - offset), 
                         (center_lon + offset, center_lat + offset), 
                         (center_lon - offset, center_lat + offset)])
        gdf = gpd.GeoDataFrame({'geometry': [polygon], 'name': [location_name]}, crs='EPSG:4326')
        st.info("✅ OSMnx Fallback erfolgreich (800m Radius)")
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
    "🌱 Urban Greening Plan",
    "🚀 What We Plan Next",
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

elif page == "🌱 Urban Greening Plan":
    st.title("🌱 Spezifischer Begrünungsplan: Landsberger Straße, München")
    st.caption("Wissenschaftlich fundierte Empfehlungen für die hochbelastete Hauptverkehrsachse zwischen Hauptbahnhof und Westend")
    
    # Standortanalyse
    st.header("1. Standortspezifische Analyse: Landsberger Straße")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Lagecharakteristik")
        st.markdown("""
        - **Lage:** Hauptausfallstraße vom Münchner Hauptbahnhof durch Schwanthalerhöhe/Westend
        - **Länge:** 6,5 km, verkehrlich hochfrequentiert (Teil der B2 ab Trappentreustraße)
        - **Umgebung:** Augustiner-Brauerei, Central Tower, ehem. Hauptzollamt, ICE-Halle
        - **Verkehrsaufkommen:** >30.000 Kfz/Tag, Straßenbahn Linie 19, hohe Abgasbelastung
        """)
    
    with col2:
        st.subheader("🌡️ Klimatische Herausforderungen")
        st.markdown("""
        - **NO₂-Belastung:** Überschreitung der 40 µg/m³ Grenzwerte an Hauptverkehrsstraßen
        - **Überwärmung:** Starke Aufheizung durch Asphalt und dichte Bebauung
        - **Windverhältnisse:** Hauptwind aus West-Südwest - ideale Belüftungsrichtung
        - **Bodenqualität:** Verdichtete, salzbelastete Böden durch Winterdienst
        """)
    
    # Wissenschaftlich begründete Baumauswahl
    st.header("2. Wissenschaftlich fundierte Baumarten-Empfehlungen")
    st.info("💡 **Auswahlkriterien:** Basierend auf Bayern LWG 'Stadtgrün 2021+' Forschung und München-spezifischen Klimadaten")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌳 Tilia cordata 'Rancho'")
        st.caption("(Kleinblättrige Linde - Bewährte Sorte)")
        with st.container():
            st.markdown("""
            **Wissenschaftliche Begründung:**
            - ✅ **Bewährt in München:** Bereits erfolgreich in der Maxvorstadt etabliert
            - ✅ **NO₂-Filter:** Nachgewiesene Luftreinigungsleistung von 27 kg/Jahr pro Baum
            - ✅ **Kühlleistung:** Bis zu 400 kWh Kühlungsäquivalent durch Transpiration
            - ✅ **Salztoleranz:** Moderate Resistenz gegen Winterstreusalz
            
            **Spezifisch für Landsberger Straße:**
            Perfekt für Abschnitte mit breiteren Gehwegen (>3m). Hohe Biomasseproduktion für maximale CO₂-Speicherung.
            """)
    
    with col2:
        st.subheader("🌳 Gleditsia triacanthos 'Skyline'")
        st.caption("(Dornenlose Honiglocke)")
        with st.container():
            st.markdown("""
            **Wissenschaftliche Begründung:**
            - ✅ **Extremstandort-tolerant:** Verträgt Hitze bis 42°C und Trockenperioden >8 Wochen
            - ✅ **Schmale Krone:** Ideal für beengte Verhältnisse der Landsberger Straße
            - ✅ **Geringe Laubmenge:** Reduziert Reinigungsaufwand bei hohem Verkehrsaufkommen
            - ✅ **Stickstoff-Fixierung:** Verbessert allmählich die Bodenqualität
            
            **Spezifisch für Landsberger Straße:**
            Optimal für enge Bereiche zwischen Augustiner-Brauerei und Hauptzollamt. Übersteht Baustellenstaub.
            """)
    
    with col3:
        st.subheader("🌳 Quercus cerris")
        st.caption("(Zerr-Eiche - Zukunftsbaum)")
        with st.container():
            st.markdown("""
            **Wissenschaftliche Begründung:**
            - ✅ **Klimawandel-resistent:** Bayern LWG Testsieger für Stadtklima 2071-2100
            - ✅ **Hohe Luftreinigung:** 48 kg Schadstoffe/Jahr bei Vollgröße
            - ✅ **Biodiversität:** Lebensraum für 200+ Insektenarten
            - ✅ **Langlebigkeit:** 150+ Jahre Standzeit bei optimaler Pflege
            
            **Spezifisch für Landsberger Straße:**
            Zukunftsinvestition für Bereiche mit ausreichend Platz. Wird steigende Temperaturen problemlos überstehen.
            """)
    
    # Unterpflanzung wissenschaftlich begründet
    st.header("3. Klimaangepasste Unterpflanzung")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌿 Stachys byzantina")
        st.caption("(Woll-Ziest)")
        st.markdown("""
        **Warum hier:** Silbrige Blätter reflektieren Hitze, extrem trockenheitstolerant. 
        Bewährt an münchner Straßenstandorten.
        """)
    
    with col2:
        st.subheader("🌿 Sedum spurium")
        st.caption("(Kaukasus-Fetthenne)")
        st.markdown("""
        **Warum hier:** Sukkulente Eigenschaften, speichert Regenwasser. 
        Verträgt Salz und Abgase ausgezeichnet.
        """)
    
    with col3:
        st.subheader("🌿 Festuca gautieri")
        st.caption("(Bärenfell-Schwingel)")
        st.markdown("""
        **Warum hier:** Immergrün, kompakt, tritt-resistent. 
        Ideal für hochfrequentierte Fußgängerbereiche.
        """)
    
    # Umsetzungsplan
    st.header("4. 🚀 Konkreter Handlungsplan")
    
    st.subheader("Phase 1: Vorbereitung (Monate 1-2)")
    st.markdown("""
    **1.1 Genehmigungen einholen:**
    - 📞 **Baureferat München:** Tel. 089/233-60001 (Straßenbegrünung)
    - 📞 **Referat für Klima- und Umweltschutz:** Tel. 089/233-47878 (Förderanträge)
    - 📋 **Erforderlich:** Straßenbaumkataster-Eintrag, Leitungsauskunft, Verkehrssicherheit
    
    **1.2 Fördermittel beantragen:**
    - 💰 **München:** Bis zu 50% Förderung für Straßenbegrünung
    - 💰 **Bayern:** KLIMAWIN-Programm für CO₂-Reduktion
    - 💰 **Bund:** Förderrichtlinie Stadtnatur 2030
    """)
    
    st.subheader("Phase 2: Planung & Partner (Monate 2-3)")
    
    # Lokale Unternehmen mit Links
    st.markdown("**🏢 Empfohlene Münchner Fachunternehmen:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Große Projekte (>50 Bäume):**
        - **[GZIMI GmbH](https://gzimi.de/)** - Spezialist für Großprojekte, Teil der idverde Gruppe
        - **[Badruk Gartengestaltung](https://www.badruk.de/)** - Familienbetrieb seit 1989, eigene Baumschule
        - **[Verde Gartenbau München](https://www.verde-gartenbau.de/)** - Meisterbetrieb, Stadtbegrünung
        """)
    
    with col2:
        st.markdown("""
        **Beratung & Planung:**
        - **[Green City e.V. - Begrünungsbüro](https://www.greencity.de/projekt/begruenungsbuero/)** - Kostenlose Erstberatung
        - **Bayerische Architektenkammer** - Zertifizierte Landschaftsarchitekten
        - **Verband Garten-, Landschafts- und Sportplatzbau Bayern e.V.** - Qualifizierte Ausführung
        """)
    
    st.subheader("Phase 3: Umsetzung (Monate 4-12)")
    st.markdown("""
    **3.1 Optimaler Pflanztermin:** Oktober-November (nach Augustiner Oktoberfest-Verkehr)
    
    **3.2 Spezielle Anforderungen Landsberger Straße:**
    - 🚧 **Verkehrsführung:** Abstimmung mit MVG (Tram 19) und Polizei
    - 🌱 **Substrat:** Strukturboden mit 40% Grobanteil für Verdichtungsresistenz  
    - 💧 **Bewässerung:** Mindestens 3 Jahre Anwachsgarantie bei Trockenheit
    - 🛡️ **Schutz:** Verstärkte Stammschutzmanschetten gegen Vandalismus
    """)
    
    # Impact-Berechnung
    st.header("5. 📊 Kalkulierte Auswirkungen für die Landsberger Straße")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🌱 CO₂-Reduktion/Jahr", 
            value="24-40 Tonnen",
            help="Bei 100 Bäumen verschiedener Arten, basierend auf LWG Bayern-Daten"
        )
        st.metric(
            label="💨 NO₂-Filterung",
            value="2.7 Tonnen/Jahr", 
            help="Besonders relevant für die hochbelastete Landsberger Straße"
        )
    
    with col2:
        st.metric(
            label="❄️ Kühlleistung",
            value="40 MWh/Jahr",
            help="Entspricht 15% Energieeinsparung für angrenzende Gebäude"
        )
        st.metric(
            label="💧 Regenwasser-Retention",
            value="80.000 L/Jahr",
            help="Entlastung der Kanalisation bei Starkregenereignissen"
        )
    
    with col3:
        st.metric(
            label="🏠 Immobilienwert-Steigerung",
            value="4-7%",
            help="Durchschnittlich für Objekte in 100m Nähe zu Straßenbäumen"
        )
        st.metric(
            label="💰 ROI-Zeitraum",
            value="6-9 Jahre",
            help="Amortisation durch Energie-/Gesundheitskosten-Einsparungen"
        )
    
    # Erfolgskontrolle
    st.header("6. 📈 Monitoring & Erfolgskontrolle")
    st.success("""
    **Empfohlene Messungen:**
    ✅ **Luftqualität:** NO₂-Passivsammler vor/nach Pflanzung  
    ✅ **Mikroklima:** Temperatur-Logger in 1m und 3m Höhe  
    ✅ **Biodiversität:** Insektenzählungen Mai-September  
    ✅ **Baumgesundheit:** Jährliches Vitalitäts-Assessment  
    ✅ **Bürgerzufriedenheit:** Umfragen zu Aufenthaltsqualität
    """)
    
    st.info("""
    💡 **Besonderheit Landsberger Straße:** Als Teil der historischen Verbindung zum Hauptbahnhof 
    und wichtige ÖPNV-Achse ist diese Begrünung ein Leuchtturmprojekt für nachhaltige Mobilität 
    in München. Die wissenschaftliche Dokumentation kann als Blaupause für andere Hauptverkehrsstraßen dienen.
    """)
    
    st.markdown("---")
    st.caption("📚 **Wissenschaftliche Grundlagen:** Bayern LWG Veitshöchheim 'Stadtgrün 2021+', München Klimafunktionskarte 2022, EU-Luftqualitätsrichtlinie 2008/50/EG")

elif page == "🚀 What We Plan Next":
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
            pass  # Falls Logo nicht vorhanden
    with col2:
        st.markdown("<h1 style='margin-bottom: 0;'>friGIS</h1>", unsafe_allow_html=True)
    
    def load_osm_data_with_retry(polygon, tags, max_retries=3):
        """OSM Daten mit Retry-Logik laden"""
        for attempt in range(max_retries):
            try:
                data = ox.features_from_polygon(polygon, tags=tags)
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"OSM Versuch {attempt + 1} fehlgeschlagen, versuche erneut...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    st.error(f"OSM Daten konnten nach {max_retries} Versuchen nicht geladen werden: {e}")
                    return gpd.GeoDataFrame()  # Leeres GeoDataFrame zurückgeben
    
    def gebaeudedichte_analysieren_und_plotten(grid, buildings, gebiet):
        if buildings.empty:
            st.warning("⚠️ Keine Gebäudedaten verfügbar - Standardwerte verwendet")
            grid["building_ratio"] = 0.1  # Standardwert
        else:
            progress = st.progress(0, text="🏗️ Calculating building density...")
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
            progress.progress(1.0, text="🏗️ Building density calculated.")
            progress.empty()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        grid.plot(ax=ax, column="building_ratio", cmap="Reds", legend=True,
                  edgecolor="grey", linewidth=0.2)
        if not buildings.empty:
            buildings.plot(ax=ax, color="lightgrey", edgecolor="black", alpha=0.5)
        gebiet.boundary.plot(ax=ax, color="blue", linewidth=1.5)
        ax.set_title("1️⃣ Building Density (Red = dense)")
        
        # SEHR ENGER Fokus - nur das tatsächlich analysierte Grid anzeigen
        grid_bounds = grid.total_bounds
        margin = 15  # Sehr kleiner Rand: nur 15m um das Grid
        ax.set_xlim(grid_bounds[0] - margin, grid_bounds[2] + margin)
        ax.set_ylim(grid_bounds[1] - margin, grid_bounds[3] + margin)
        ax.axis("equal")
        plt.tight_layout()
        return fig

    def distanz_zu_gruenflaechen_analysieren_und_plotten(grid, greens, gebiet, max_dist=500):
        if greens.empty:
            st.warning("⚠️ Keine Grünflächendaten verfügbar - Standardwerte verwendet")
            grid["dist_to_green"] = max_dist
            grid["score_distance_norm"] = 1.0
        else:
            progress = st.progress(0, text="🌳 Calculating distance to green areas...")
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
                progress.progress(1.0, text="🌳 Distance to green calculated.")
                progress.empty()
            except Exception as e:
                st.warning(f"Fehler bei Grünflächenanalyse: {e}")
                grid["dist_to_green"] = max_dist
                grid["score_distance_norm"] = 1.0
        
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
        
        # SEHR ENGER Fokus - nur das tatsächlich analysierte Grid anzeigen  
        grid_bounds = grid.total_bounds
        margin = 15  # Sehr kleiner Rand: nur 15m um das Grid
        ax.set_xlim(grid_bounds[0] - margin, grid_bounds[2] + margin)
        ax.set_ylim(grid_bounds[1] - margin, grid_bounds[3] + margin)
        ax.axis("equal")
        plt.tight_layout()
        return fig

    def heatmap_mit_temperaturdifferenzen(ort_name, jahr=2022, radius_km=2.0, resolution_km=0.7):
        """ERWEITERTE Temperaturdaten - MEHR Punkte"""
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
        lats = np.arange(lat0 - radius_km / 111, lat0 + radius_km / 111 + 1e-6, resolution_km / 111)
        lons = np.arange(lon0 - radius_km / 85, lon0 + radius_km / 85 + 1e-6, resolution_km / 85)
    
        punkt_daten = []
        ref_temp = None
        total_points = len(lats) * len(lons)
        progress = st.progress(0, text=f"🔄 Temperaturdaten werden geladen... ({total_points} Punkte)")
        count = 0
        
        def fetch_temperature(lat, lon):
            for _ in range(2):  # Reduziert auf 2 Versuche
                try:
                    url = (
                        f"https://archive-api.open-meteo.com/v1/archive?"
                        f"latitude={lat}&longitude={lon}"
                        f"&start_date={jahr}-06-01&end_date={jahr}-08-31"
                        f"&daily=temperature_2m_max&timezone=auto"
                    )
                    r = session.get(url, timeout=8)  # Reduziert auf 8s
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
        
        # Erhöhte Parallelität für OPTIMIERTE Anzahl Temperaturpunkte
        coords = [(lat, lon) for lat in lats for lon in lons]
        with ThreadPoolExecutor(max_workers=6) as executor:  # Reduziert von 8 auf 6
            futures = [executor.submit(fetch_temperature, lat, lon) for lat, lon in coords]
            
            for future in as_completed(futures):
                lat, lon, temp = future.result()
                if temp is not None:
                    punkt_daten.append([lat, lon, temp])
                    
                    if abs(lat - lat0) < resolution_km / 222 and abs(lon - lon0) < resolution_km / 170:
                        ref_temp = temp
                
                count += 1
                progress.progress(min(count / total_points, 1.0), 
                               text=f"🔄 Temperaturdaten werden geladen... ({count}/{total_points})")
    
        progress.empty()
    
        if not punkt_daten:
            st.warning("⚠️ Nicht genug Temperaturdaten verfügbar.")
            return None
            
        if ref_temp is None:
            ref_temp = np.mean([temp for _, _, temp in punkt_daten])
            st.info("ℹ️ Mittelpunktwert geschätzt")
    
        differenzpunkte = [
            [lat, lon, round(temp - ref_temp, 2)]
            for lat, lon, temp in punkt_daten
        ]
    
        # Verbesserte Heatmap mit MEHR Datenpunkten
        m = folium.Map(location=[lat0, lon0], zoom_start=13, tiles="CartoDB positron")
        HeatMap(
            [[lat, lon, abs(diff)] for lat, lon, diff in differenzpunkte],
            radius=22,  # Größerer Radius für bessere Sichtbarkeit
            blur=20,    # Optimierter Blur
            max_zoom=13,
            gradient={0.0: "green", 0.3: "lightyellow", 0.6: "orange", 1.0: "red"}
        ).add_to(m)
    
        for lat, lon, diff in differenzpunkte:
            sign = "+" if diff > 0 else ("−" if diff < 0 else "±")
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(html=f"<div style='font-size:10pt; color:black'><b>{sign}{abs(diff):.2f}°C</b></div>")
            ).add_to(m)
    
        st.success(f"✅ {len(punkt_daten)} Temperaturpunkte geladen (OPTIMIERT: {radius_km}km Radius, {resolution_km}km Auflösung = ~{len(punkt_daten)} Messpunkte)!")
        return m
    
    def analysiere_reflektivitaet_graustufen(stadtteil_name, n_clusters=5, year_range="2020-01-01/2024-12-31"):
        try:
            progress = st.progress(0, text="🔍 Satellitendaten werden gesucht...")
            
            gebiet = geocode_to_gdf_with_fallback(stadtteil_name)
            if gebiet is None:
                st.warning("❌ Gebiet konnte nicht gefunden werden.")
                progress.empty()
                return None
            
            # VIEL größerer Radius für k-Means Satellitendaten
            bounds = gebiet.total_bounds
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            large_offset = 0.015  # Viel größerer Radius: ca. 1.5km statt 350m
            large_polygon = Polygon([
                (center_lon - large_offset, center_lat - large_offset),
                (center_lon + large_offset, center_lat - large_offset), 
                (center_lon + large_offset, center_lat + large_offset),
                (center_lon - large_offset, center_lat + large_offset)
            ])
            large_gebiet = gpd.GeoDataFrame({'geometry': [large_polygon]}, crs='EPSG:4326')
            bbox = large_gebiet.total_bounds
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
        
            # VIEL bessere Auflösung für k-Means
            stack = stackstac.stack(
                [item],
                assets=["B04", "B03", "B02"],
                resolution=5,  # Deutlich verbessert von 10 auf 5
                bounds_latlon=bbox.tolist(),
                epsg=utm_crs
            )
            rgb = stack.isel(band=[0,1,2], time=0).transpose("y","x","band").values
            rgb = np.nan_to_num(rgb)
            rgb_scaled = np.clip((rgb / 3000) * 255, 0, 255).astype(np.uint8)
        
            h, w, _ = rgb_scaled.shape
            pixels = rgb_scaled.reshape(-1, 3)
            progress.progress(0.7, text="🔢 k-Means Clustering wird durchgeführt...")
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

        # OSM Daten mit Retry-Logik
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

        # Grid erstellen - HÖHERE Auflösung
        cell_size = 40  # Reduziert von 50 auf 40 für höhere Auflösung
        minx, miny, maxx, maxy = area.bounds
        grid_cells = [
            box(x, y, x + cell_size, y + cell_size)
            for x in np.arange(minx, maxx, cell_size)
            for y in np.arange(miny, maxy, cell_size)
            if box(x, y, x + cell_size, y + cell_size).intersects(area)
        ]
        grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)

        # Analysen durchführen (mit Fehlerbehandlung)
        try:
            st.subheader("Gebäudedichte")
            fig1 = gebaeudedichte_analysieren_und_plotten(grid.copy(), buildings, gebiet)
            st.pyplot(fig1)
            plt.close(fig1)  # Memory-Management
        except Exception as e:
            st.error(f"Gebäudedichte-Analyse fehlgeschlagen: {e}")

        try:
            st.subheader("Distanz zu Grünflächen")
            fig2 = distanz_zu_gruenflaechen_analysieren_und_plotten(grid.copy(), greens, gebiet)
            st.pyplot(fig2)
            plt.close(fig2)  # Memory-Management
        except Exception as e:
            st.error(f"Grünflächen-Analyse fehlgeschlagen: {e}")

        try:
            st.subheader("Temperaturdifferenz Heatmap")
            heatmap = heatmap_mit_temperaturdifferenzen(ort_name=stadtteil)
            if heatmap:
                st.components.v1.html(heatmap._repr_html_(), height=600)
        except Exception as e:
            st.error(f"Temperatur-Analyse fehlgeschlagen: {e}")

        try:
            st.subheader("k-Means Clusteranalyse von Satellitendaten")
            fig3 = analysiere_reflektivitaet_graustufen(stadtteil, n_clusters=5)
            if fig3:
                st.pyplot(fig3)
                plt.close(fig3)  # Memory-Management
        except Exception as e:
            st.error(f"Satellitendaten-Analyse fehlgeschlagen: {e}")

        # Am Ende der Analyse
        st.session_state.analysis_complete = True
        st.success("✅ Analyse abgeschlossen! Du kannst jetzt eine neue Analyse starten.")
    
    # Call main function when on the main app page
    main()
