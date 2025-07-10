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
    "📊 Analysis Methods Info",
    "🌱 Urban Greening Plan",
    "🚀 What We Plan Next",
    "🐞 Report a Bug"
])

if page == "📊 Analysis Methods Info":
    st.title("📊 friGIS Analysis Methods")
    st.markdown("Comprehensive overview of all analytical methods used in our urban heat analysis platform")
    
    # Tabs für die verschiedenen Methoden
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Building Density", "🌳 Distance to Green", "🔥 Temperature Heatmap", "🛰️ Satellite k-Means"])
    
    with tab1:
        st.header("🏗️ Building Density Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **How it works:**
            Building footprints from OpenStreetMap are used to calculate the ratio of built area per cell in our analysis grid.
            
            **Technical Details:**
            - Grid cell size: 40m × 40m for high resolution
            - Data source: OpenStreetMap building polygons
            - Calculation: Building area ÷ Cell area = Building ratio
            - Color coding: Red indicates high density (heat accumulation zones)
            
            **Why it's useful:**
            High building density often correlates with heat accumulation in cities. This metric helps identify particularly heat-stressed urban zones that would benefit most from greening interventions.
            """)
        with col2:
            st.info("""
            **Key Insights:**
            - Red areas = High heat risk
            - Urban canyons trap heat
            - Guides tree placement priorities
            - Identifies cooling needs
            """)
    
    with tab2:
        st.header("🌳 Distance to Green Spaces")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **How it works:**
            We calculate the distance from each urban grid cell centroid to the nearest green space (parks, gardens, forests).
            
            **Technical Details:**
            - Green space data: OpenStreetMap (leisure=park, landuse=forest, etc.)
            - Distance calculation: Euclidean distance to nearest green polygon
            - Maximum distance: 500m (beyond this, cooling effect is minimal)
            - Color coding: Red = far from green, Green = close to nature
            
            **Why it's important:**
            Proximity to green areas directly influences local cooling and microclimates. Areas far from green spaces are more heat-prone and should be prioritized for new tree plantings.
            """)
        with col2:
            st.info("""
            **Key Insights:**
            - <100m = Excellent cooling
            - 100-300m = Moderate cooling  
            - >300m = Heat island risk
            - Guides intervention zones
            """)
    
    with tab3:
        st.header("🔥 Temperature Heatmap Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **How it works:**
            Historical temperature data (Open-Meteo API) is collected across a grid of points. Temperature differences from the central reference point show relative heating patterns.
            
            **Technical Details:**
            - Data source: Open-Meteo historical weather archive
            - Time period: Summer months (June-August) for maximum heat stress
            - Grid resolution: 0.7km spacing for detailed coverage
            - Reference: Central point temperature as baseline
            - Visualization: Folium heatmap with temperature difference markers
            
            **Why it's valuable:**
            Real temperature data helps identify actual hotspots and temperature variations within neighborhoods, validating where cooling interventions are most needed.
            """)
        with col2:
            st.info("""
            **Key Insights:**
            - Red zones = Hotspots (+1-3°C)
            - Blue zones = Cool spots
            - Validates heat island effects
            - Quantifies intervention impact
            """)
    
    with tab4:
        st.header("🛰️ Satellite k-Means Clustering")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **How it works:**
            Sentinel-2 satellite imagery is analyzed using k-Means clustering to group pixels by brightness/reflectivity, which correlates with surface heating potential.
            
            **Technical Details:**
            - Data source: Microsoft Planetary Computer (Sentinel-2 L2A)
            - Resolution: 5m per pixel for detailed surface analysis
            - Spectral bands: RGB (B04, B03, B02) for visible light analysis
            - Clustering: 5 brightness categories from dark (heat-absorbing) to bright (heat-reflecting)
            - Coverage area: 1.5km radius for comprehensive analysis
            
            **Scientific basis:**
            Dark surfaces (asphalt, dark roofs) absorb heat, while bright surfaces (vegetation, light materials) reflect heat. This analysis identifies which areas have the highest heating potential.
            """)
        with col2:
            st.info("""
            **Key Insights:**
            - Dark clusters = High heat absorption
            - Bright clusters = Heat reflection
            - Identifies material types
            - Validates surface interventions
            """)
    
    # Zusammenfassung
    st.header("🔬 Integrated Analysis Approach")
    st.success("""
    **Why combine all four methods?**
    
    ✅ **Building Density** shows structural heat trapping  
    ✅ **Distance to Green** reveals cooling deficit zones  
    ✅ **Temperature Data** provides real-world validation  
    ✅ **Satellite Analysis** identifies surface material impacts  
    
    Together, these create a comprehensive picture of urban heat patterns and optimal intervention strategies.
    """)

elif page == "🌱 Urban Greening Plan":
    # Sprachauswahl oben
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🇩🇪 Deutsch", type="secondary"):
            st.session_state.greening_language = "de"
    with col2:
        if st.button("🇬🇧 English", type="secondary"):
            st.session_state.greening_language = "en"
    
    # Standardsprache setzen falls nicht vorhanden
    if 'greening_language' not in st.session_state:
        st.session_state.greening_language = "de"
    
    # Deutsche Version
    if st.session_state.greening_language == "de":
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

    # Englische Version
    else:
        st.title("🌱 Specific Greening Plan: Landsberger Straße, Munich")
        st.caption("Science-based recommendations for the highly trafficked main arterial between Central Station and Westend")
        
        # Site Analysis
        st.header("1. Site-Specific Analysis: Landsberger Straße")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Location Characteristics")
            st.markdown("""
            - **Location:** Main arterial from Munich Central Station through Schwanthalerhöhe/Westend
            - **Length:** 6.5 km, heavy traffic (part of B2 from Trappentreustraße)
            - **Surroundings:** Augustiner Brewery, Central Tower, former Main Customs Office, ICE Hall
            - **Traffic Volume:** >30,000 vehicles/day, Tram Line 19, high emission levels
            """)
        
        with col2:
            st.subheader("🌡️ Climate Challenges")
            st.markdown("""
            - **NO₂ Pollution:** Exceeding 40 µg/m³ limits on main traffic arteries
            - **Heat Island Effect:** Strong heating through asphalt and dense construction
            - **Wind Patterns:** Prevailing winds from west-southwest - ideal ventilation direction
            - **Soil Quality:** Compacted, salt-contaminated soils from winter road maintenance
            """)
        
        # Science-based tree selection
        st.header("2. Science-Based Tree Species Recommendations")
        st.info("💡 **Selection Criteria:** Based on Bavaria LWG 'Urban Green 2021+' research and Munich-specific climate data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌳 Tilia cordata 'Rancho'")
            st.caption("(Small-Leaved Lime - Proven Variety)")
            with st.container():
                st.markdown("""
                **Scientific Rationale:**
                - ✅ **Proven in Munich:** Successfully established in Maxvorstadt district
                - ✅ **NO₂ Filter:** Proven air cleaning performance of 27 kg/year per tree
                - ✅ **Cooling Power:** Up to 400 kWh cooling equivalent through transpiration
                - ✅ **Salt Tolerance:** Moderate resistance to winter road salt
                
                **Specific to Landsberger Straße:**
                Perfect for sections with wider sidewalks (>3m). High biomass production for maximum CO₂ storage.
                """)
        
        with col2:
            st.subheader("🌳 Gleditsia triacanthos 'Skyline'")
            st.caption("(Thornless Honey Locust)")
            with st.container():
                st.markdown("""
                **Scientific Rationale:**
                - ✅ **Extreme Site Tolerant:** Withstands heat up to 42°C and drought periods >8 weeks
                - ✅ **Narrow Crown:** Ideal for confined conditions of Landsberger Straße
                - ✅ **Low Leaf Litter:** Reduces maintenance burden with high traffic volume
                - ✅ **Nitrogen Fixation:** Gradually improves soil quality
                
                **Specific to Landsberger Straße:**
                Optimal for tight spaces between Augustiner Brewery and Main Customs Office. Survives construction dust.
                """)
        
        with col3:
            st.subheader("🌳 Quercus cerris")
            st.caption("(Turkey Oak - Future Tree)")
            with st.container():
                st.markdown("""
                **Scientific Rationale:**
                - ✅ **Climate Change Resistant:** Bavaria LWG test winner for urban climate 2071-2100
                - ✅ **High Air Purification:** 48 kg pollutants/year at full size
                - ✅ **Biodiversity:** Habitat for 200+ insect species
                - ✅ **Longevity:** 150+ years lifespan with optimal care
                
                **Specific to Landsberger Straße:**
                Future investment for areas with sufficient space. Will easily handle rising temperatures.
                """)
        
        # Climate-adapted understory
        st.header("3. Climate-Adapted Understory Planting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌿 Stachys byzantina")
            st.caption("(Lamb's Ear)")
            st.markdown("""
            **Why here:** Silver leaves reflect heat, extremely drought tolerant. 
            Proven at Munich street locations.
            """)
        
        with col2:
            st.subheader("🌿 Sedum spurium")
            st.caption("(Caucasian Stonecrop)")
            st.markdown("""
            **Why here:** Succulent properties, stores rainwater. 
            Excellently tolerates salt and exhaust fumes.
            """)
        
        with col3:
            st.subheader("🌿 Festuca gautieri")
            st.caption("(Bear Skin Fescue)")
            st.markdown("""
            **Why here:** Evergreen, compact, foot-traffic resistant. 
            Ideal for high-frequency pedestrian areas.
            """)
        
        # Implementation plan
        st.header("4. 🚀 Concrete Action Plan")
        
        st.subheader("Phase 1: Preparation (Months 1-2)")
        st.markdown("""
        **1.1 Obtain Permits:**
        - 📞 **Munich Building Department:** Tel. 089/233-60001 (Street greening)
        - 📞 **Climate & Environmental Protection Department:** Tel. 089/233-47878 (Funding applications)
        - 📋 **Required:** Street tree registry entry, utility clearance, traffic safety approval
        
        **1.2 Apply for Funding:**
        - 💰 **Munich:** Up to 50% funding for street greening
        - 💰 **Bavaria:** KLIMAWIN program for CO₂ reduction
        - 💰 **Federal:** Urban Nature 2030 funding directive
        """)
        
        st.subheader("Phase 2: Planning & Partners (Months 2-3)")
        
        st.markdown("**🏢 Recommended Munich Specialist Companies:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Large Projects (>50 trees):**
            - **[GZIMI GmbH](https://gzimi.de/)** - Specialist for large projects, part of idverde Group
            - **[Badruk Garden Design](https://www.badruk.de/)** - Family business since 1989, own nursery
            - **[Verde Garden Construction Munich](https://www.verde-gartenbau.de/)** - Master craftsman, urban greening
            """)
        
        with col2:
            st.markdown("""
            **Consulting & Planning:**
            - **[Green City e.V. - Greening Office](https://www.greencity.de/projekt/begruenungsbuero/)** - Free initial consultation
            - **Bavarian Chamber of Architects** - Certified landscape architects
            - **Association of Garden, Landscape and Sports Ground Construction Bavaria** - Qualified execution
            """)
        
        st.subheader("Phase 3: Implementation (Months 4-12)")
        st.markdown("""
        **3.1 Optimal Planting Time:** October-November (after Augustiner Oktoberfest traffic)
        
        **3.2 Special Requirements Landsberger Straße:**
        - 🚧 **Traffic Management:** Coordination with MVG (Tram 19) and Police
        - 🌱 **Substrate:** Structural soil with 40% coarse fraction for compaction resistance
        - 💧 **Irrigation:** Minimum 3-year establishment guarantee during drought
        - 🛡️ **Protection:** Reinforced trunk protection sleeves against vandalism
        """)
        
        # Impact calculation
        st.header("5. 📊 Calculated Impact for Landsberger Straße")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🌱 CO₂ Reduction/Year", 
                value="24-40 tonnes",
                help="For 100 trees of various species, based on LWG Bavaria data"
            )
            st.metric(
                label="💨 NO₂ Filtering",
                value="2.7 tonnes/year", 
                help="Particularly relevant for the heavily polluted Landsberger Straße"
            )
        
        with col2:
            st.metric(
                label="❄️ Cooling Power",
                value="40 MWh/year",
                help="Equivalent to 15% energy savings for adjacent buildings"
            )
            st.metric(
                label="💧 Rainwater Retention",
                value="80,000 L/year",
                help="Stormwater system relief during heavy rain events"
            )
        
        with col3:
            st.metric(
                label="🏠 Property Value Increase",
                value="4-7%",
                help="Average for properties within 100m of street trees"
            )
            st.metric(
                label="💰 ROI Period",
                value="6-9 years",
                help="Payback through energy/health cost savings"
            )
        
        # Success monitoring
        st.header("6. 📈 Monitoring & Success Control")
        st.success("""
        **Recommended Measurements:**
        ✅ **Air Quality:** NO₂ passive samplers before/after planting  
        ✅ **Microclimate:** Temperature loggers at 1m and 3m height  
        ✅ **Biodiversity:** Insect counts May-September  
        ✅ **Tree Health:** Annual vitality assessment  
        ✅ **Citizen Satisfaction:** Surveys on quality of stay
        """)
        
        st.info("""
        💡 **Special Feature Landsberger Straße:** As part of the historic connection to Central Station 
        and important public transport axis, this greening is a flagship project for sustainable mobility 
        in Munich. Scientific documentation can serve as blueprint for other main traffic arteries.
        """)
        
        st.markdown("---")
        st.caption("📚 **Scientific Basis:** Bavaria LWG Veitshöchheim 'Urban Green 2021+', Munich Climate Function Map 2022, EU Air Quality Directive 2008/50/EC")

elif page == "🚀 What We Plan Next":
    st.title("🚀 What We Plan Next")
    st.caption("Our vision for comprehensive urban cooling solutions and data-driven monitoring")
    
    # Vision Overview
    st.header("🌍 Our Vision for Scalable Urban Cooling")
    st.markdown("""
    We're building the next generation of urban climate analysis tools that go far beyond our current prototype. 
    Our goal is to create customized cooling strategies for any location worldwide, integrating multiple 
    technologies and providing real-time monitoring capabilities.
    """)
    
    # Expanded Greening Plans
    st.header("1. 🌱 Universal Greening Plans")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Location-Specific Customization")
        st.markdown("""
        **Global Coverage:**
        - Generate tailored greening recommendations for any city or neighborhood worldwide
        - Adapt plant selections based on local climate, soil conditions, and regulations
        - Account for regional precipitation patterns, temperature ranges, and seasonal variations
        
        **Flexible Area Types:**
        - Streets and transportation corridors
        - Public squares and pedestrian zones
        - Commercial districts and business areas
        - Residential neighborhoods
        - Industrial zones requiring specialized approaches
        - School grounds and educational institutions
        """)
    
    with col2:
        st.subheader("⚙️ Advanced Customization Options")
        st.markdown("""
        **User-Defined Parameters:**
        - Budget constraints and funding sources
        - Maintenance capacity and long-term care
        - Aesthetic preferences and community input
        - Traffic patterns and pedestrian flow
        - Underground infrastructure limitations
        - Local wildlife and biodiversity goals
        
        **Integration with Urban Planning:**
        - Coordination with existing city development plans
        - Compliance with local environmental regulations
        - Integration with smart city infrastructure
        """)
    
    # Multi-Technology Approach
    st.header("2. 🏗️ Beyond Greening: Comprehensive Cooling Technologies")
    
    st.info("💡 **Holistic Approach:** Our future cooling plans will integrate multiple proven technologies for maximum impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🛣️ Advanced Surface Materials")
        st.markdown("""
        **Cool Pavements:**
        - Light-colored asphalt with high solar reflectance
        - Permeable concrete for stormwater management
        - Phase-change materials for temperature regulation
        
        **Reflective Coatings:**
        - Cool roof technologies and materials
        - Highly reflective street markings
        - Solar-reflective sidewalk treatments
        """)
    
    with col2:
        st.subheader("🏢 Building Envelope Solutions")
        st.markdown("""
        **Facade Technologies:**
        - Cool wall paints and coatings
        - Green facades and living walls
        - Shading systems and architectural features
        
        **Integrated Systems:**
        - Building-integrated photovoltaics with cooling
        - Natural ventilation enhancement
        - Thermal mass optimization
        """)
    
    with col3:
        st.subheader("💧 Water-Based Cooling")
        st.markdown("""
        **Active Cooling:**
        - Misting systems for public spaces
        - Water features and fountains
        - Evaporative cooling installations
        
        **Passive Solutions:**
        - Bioswales and rain gardens
        - Constructed wetlands
        - Integrated stormwater management
        """)
    
    # Advanced Monitoring & Data Collection
    st.header("3. 📊 Precision Monitoring & Data Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Real-Time Environmental Monitoring")
        st.markdown("""
        **Proprietary Sensor Networks:**
        - High-precision temperature and humidity sensors
        - Air quality monitoring (PM2.5, NO₂, O₃)
        - Solar radiation and UV index measurement
        - Wind speed and direction tracking
        - Soil moisture and plant health indicators
        
        **IoT Integration:**
        - Wireless sensor networks with long battery life
        - Real-time data transmission to cloud platforms
        - Edge computing for immediate local analysis
        - Integration with existing city sensor infrastructure
        """)
    
    with col2:
        st.subheader("📈 Advanced Analytics & Modeling")
        st.markdown("""
        **Quantitative Impact Assessment:**
        - Precise cooling effectiveness calculations (°C reduction)
        - Energy savings quantification for nearby buildings
        - Air quality improvement measurements
        - Carbon sequestration and emission reduction tracking
        
        **Predictive Modeling:**
        - Machine learning for optimal intervention timing
        - Climate change adaptation scenario planning
        - Long-term performance forecasting
        - Cost-benefit analysis automation
        """)
    
    # Implementation Timeline
    st.header("4. 🗓️ Development Roadmap")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📅 Phase 1: 2025")
        st.markdown("""
        **Platform Expansion:**
        - Global location database integration
        - Multi-technology recommendation engine
        - Advanced customization interface
        - Beta testing with select cities
        
        **Initial Partnerships:**
        - Pilot projects in 3-5 cities
        - Collaboration with urban planning firms
        - Integration with municipal databases
        """)
    
    with col2:
        st.subheader("📅 Phase 2: 2026")
        st.markdown("""
        **Technology Integration:**
        - Material science database expansion
        - Cost-optimization algorithms
        - Real-time monitoring system deployment
        - Mobile app for citizen engagement
        
        **Scale-Up:**
        - 20+ city implementations
        - Industry partnerships
        - Academic research collaborations
        """)
    
    with col3:
        st.subheader("📅 Phase 3: 2027+")
        st.markdown("""
        **Full Platform Launch:**
        - Global availability and localization
        - AI-powered optimization
        - Comprehensive impact tracking
        - Policy recommendation tools
        
        **Long-term Vision:**
        - Integration with climate adaptation strategies
        - Support for international climate goals
        - Open-source components for research
        """)
    
    # Partnerships & Collaboration
    st.header("5. 🤝 Strategic Partnerships & Advisory Network")
    
    st.success("""
    **Current Collaborations:**
    We are actively engaged with leading cities and innovative companies to ensure our solutions 
    meet real-world needs and can be effectively implemented at scale.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏛️ Municipal Partnerships")
        st.markdown("""
        **City Collaboration Areas:**
        - Urban planning and climate adaptation strategies
        - Data sharing and integration protocols  
        - Regulatory compliance and permitting processes
        - Budget planning and funding mechanisms
        - Community engagement and public acceptance
        - Long-term maintenance and monitoring programs
        
        **Knowledge Exchange:**
        - Best practices sharing across cities
        - Standardized impact measurement protocols
        - Policy framework development
        """)
    
    with col2:
        st.subheader("🏢 Industry Advisory Board")
        st.markdown("""
        **Corporate Expertise:**
        - Technology providers and material manufacturers
        - Construction and landscaping industry leaders
        - Environmental consulting firms
        - Smart city technology companies
        - Climate data and analytics providers
        - Sustainability certification organizations
        
        **Innovation Pipeline:**
        - Emerging technology evaluation
        - Market viability assessment
        - Implementation feasibility studies
        """)
    
    # Call to Action
    st.header("6. 📞 Get Involved")
    
    st.info("""
    **🌟 Interested in Collaboration?**
    
    Whether you're representing a city government, environmental organization, technology company, 
    or research institution, we welcome partnerships that can help accelerate urban cooling solutions.
    
    **Contact us:** julius.dickmann@muenchen.enactus.team
    
    **Areas of Interest:**
    - Pilot city partnerships for testing and validation
    - Technology integration and sensor deployment
    - Research collaborations and data sharing
    - Funding and investment opportunities
    - Policy development and regulatory frameworks
    """)
    
    st.markdown("---")
    st.caption("🌱 **Mission:** Democratizing access to science-based urban cooling solutions for cities worldwide, contributing to global climate resilience and improved quality of life for urban populations.")

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
        """Load OSM data with retry logic"""
        for attempt in range(max_retries):
            try:
                data = ox.features_from_polygon(polygon, tags=tags)
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"OSM attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    st.error(f"OSM data could not be loaded after {max_retries} attempts: {e}")
                    return gpd.GeoDataFrame()  # Return empty GeoDataFrame
    
    def gebaeudedichte_analysieren_und_plotten(grid, buildings, gebiet):
        if buildings.empty:
            st.warning("⚠️ No building data available - using default values")
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
            st.warning("⚠️ No green space data available - using default values")
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
                st.warning(f"Error in green space analysis: {e}")
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
        """EXTENDED Temperature data - MORE points"""
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        try:
            results = geocoder.geocode(ort_name, no_annotations=1)
        except Exception as e:
            st.error(f"🌍 Geocoding failed: {e}")
            return None

        if not results:
            st.warning("❗ Location could not be found.")
            return None
    
        lat0, lon0 = results[0]['geometry']['lat'], results[0]['geometry']['lng']
        lats = np.arange(lat0 - radius_km / 111, lat0 + radius_km / 111 + 1e-6, resolution_km / 111)
        lons = np.arange(lon0 - radius_km / 85, lon0 + radius_km / 85 + 1e-6, resolution_km / 85)
    
        punkt_daten = []
        ref_temp = None
        total_points = len(lats) * len(lons)
        progress = st.progress(0, text=f"🔄 Loading temperature data... ({total_points} points)")
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
        
        # Optimized number of parallel temperature requests
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
                               text=f"🔄 Loading temperature data... ({count}/{total_points})")
    
        progress.empty()
    
        if not punkt_daten:
            st.warning("⚠️ Not enough temperature data available.")
            return None
            
        if ref_temp is None:
            ref_temp = np.mean([temp for _, _, temp in punkt_daten])
            st.info("ℹ️ Reference temperature estimated")
    
        differenzpunkte = [
            [lat, lon, round(temp - ref_temp, 2)]
            for lat, lon, temp in punkt_daten
        ]
    
        # Enhanced Heatmap with MORE data points
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
    
        st.success(f"✅ {len(punkt_daten)} temperature points loaded (OPTIMIZED: {radius_km}km radius, {resolution_km}km resolution = ~{len(punkt_daten)} measurement points)!")
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

        stadtteil = st.text_input("🏙️ Enter district name", value="Maxvorstadt, München")

        # Button Logic mit Session State
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

        # Analyse nur ausführen wenn gestartet
        if not st.session_state.analysis_started or not stadtteil:
            return

        # Status anzeigen
        if not st.session_state.analysis_complete:
            st.info("🔄 Analysis running...")

        try:
            gebiet = geocode_to_gdf_with_fallback(stadtteil)
            if gebiet is None:
                st.error("📍 Area could not be found.")
                st.session_state.analysis_started = False
                return
                
        except Exception as e:
            st.error(f"📍 Unexpected error: {e}")
            st.session_state.analysis_started = False
            return

        polygon = gebiet.geometry.iloc[0]
        utm_crs = gebiet.estimate_utm_crs()
        gebiet = gebiet.to_crs(utm_crs)
        area = gebiet.geometry.iloc[0].buffer(0)

        # OSM data with retry logic
        tags_buildings = {"building": True}
        tags_green = {
            "leisure": ["park", "garden"],
            "landuse": ["grass", "meadow", "forest"],
            "natural": ["wood", "tree_row", "scrub"]
        }
        
        st.info("📡 Loading OSM data...")
        buildings = load_osm_data_with_retry(polygon, tags_buildings)
        greens = load_osm_data_with_retry(polygon, tags_green)
        
        # Data cleaning
        if not buildings.empty:
            buildings = buildings.to_crs(utm_crs)
            buildings = buildings[buildings.geometry.is_valid & ~buildings.geometry.is_empty]
        if not greens.empty:
            greens = greens.to_crs(utm_crs)
            greens = greens[greens.geometry.is_valid & ~greens.geometry.is_empty]

        # Create grid - HIGHER resolution
        cell_size = 40  # Reduced from 50 to 40 for higher resolution
        minx, miny, maxx, maxy = area.bounds
        grid_cells = [
            box(x, y, x + cell_size, y + cell_size)
            for x in np.arange(minx, maxx, cell_size)
            for y in np.arange(miny, maxy, cell_size)
            if box(x, y, x + cell_size, y + cell_size).intersects(area)
        ]
        grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=utm_crs)

        # Perform analyses (with error handling)
        try:
            st.subheader("Building Density")
            fig1 = gebaeudedichte_analysieren_und_plotten(grid.copy(), buildings, gebiet)
            st.pyplot(fig1)
            plt.close(fig1)  # Memory-Management
        except Exception as e:
            st.error(f"Building density analysis failed: {e}")

        try:
            st.subheader("Distance to Green Spaces")
            fig2 = distanz_zu_gruenflaechen_analysieren_und_plotten(grid.copy(), greens, gebiet)
            st.pyplot(fig2)
            plt.close(fig2)  # Memory-Management
        except Exception as e:
            st.error(f"Green space analysis failed: {e}")

        try:
            st.subheader("Temperature Difference Heatmap")
            heatmap = heatmap_mit_temperaturdifferenzen(ort_name=stadtteil)
            if heatmap:
                st.components.v1.html(heatmap._repr_html_(), height=600)
        except Exception as e:
            st.error(f"Temperature analysis failed: {e}")

        try:
            st.subheader("k-Means Cluster Analysis of Satellite Data")
            fig3 = analysiere_reflektivitaet_graustufen(stadtteil, n_clusters=5)
            if fig3:
                st.pyplot(fig3)
                plt.close(fig3)  # Memory-Management
        except Exception as e:
            st.error(f"Satellite data analysis failed: {e}")

        # At the end of analysis
        st.session_state.analysis_complete = True
        st.success("✅ Analysis completed! You can now start a new analysis.")
    
    # Call main function when on the main app page
    main()
