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

    st.subheader("Temperaturdifferenz Heatmap")
    heatmap = heatmap_mit_temperaturdifferenzen(ort_name=stadtteil)
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
