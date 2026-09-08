# Sheep Mountain terrain provenance

The demo now intersects geology with a **real cached USGS elevation grid**. It contains 416.10 m of relief across the selected 6.059 × 4.139 km map crop. Terrain is fixed across hypotheses; the model does not optimize it to match the geological labels.

## Source and cache

The source is the official [USGS 3DEP Bare Earth DEM Dynamic Image Service](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer), retrieved September 8, 2026. The service combines elevation products at multiple resolutions. USGS states that [3DEP products are freely available without use restrictions](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services). This does not change the separate rights or provenance of the user-supplied geological figure.

`terrain_source.tif` is the actual single-band float32 export, with no hillshade rendering rule. `terrain_export.json` records the returned geographic extent and download URL; that temporary URL may expire. `terrain_service.json` preserves service metadata. `terrain.json` stores the reproducible export request, raster hash and registration details. The export uses EPSG:4326, a 256 × 256 grid, bilinear sampling and `adjustAspectRatio=false`.

## Approximate registration

The supplied 697 × 671 figure has printed geographic ticks. Actual longitude tick strokes were manually read at source x = 64, 179, 294, 410 and 529 pixels for 108°12′W, 10′W, 08′W, 06′W and 04′W respectively. Latitude strokes at y = 156 and 335 correspond to 44°38′N and 44°36′N. These are stroke positions, not text-label centers.

Separate linear longitude(x) and latitude(y) fits assume north is vertical, consistent with the figure's north arrow. The source crop remains x20..286, y50..250. Its geographic bounds are:

| West | South | East | North |
| --- | --- | --- | --- |
| −108.212341752 | 44.615828678 | −108.135974603 | 44.653072626 |

WGS84 radii of curvature at the crop midpoint convert these spans to local east/north kilometers. This replaces the initial rough 52 pixels/km scale-bar estimate; **observed labels and masks are unchanged**. Old flat-surface recordings use a different scene and have incomparable scores.

This is approximate registration of a published figure, not surveyed ground control. The longitude tick-fit maximum residual is 1.59 source pixels; it does **not** establish external positional accuracy. Two latitude ticks cannot measure vertical fit residuals. The original horizontal datum, publication distortion, tick-reading uncertainty and generalized mapped contacts remain unresolved. A full-map context hillshade showed the corresponding NW–SE ridge and Bighorn River canyon, providing a qualitative location check. No registration parameter was fitted to geological-model output or target labels.

## Numerical contract and reproduction

`terrain.npz` contains `heights_km` (float32, 256²), `elevation_m`, and `xs`/`ys` identical to `target.npz`: cell centers, columns west→east, rows north→south. Sample spacing is 23.67 m east–west and 16.17 m north–south; this is export spacing, not a native-resolution or accuracy claim.

Cached service elevations range from 1129.736084 to 1545.834839 m. The model subtracts the minimum, so terrain heights range from 0 to 0.416099 km. The composite export's source vertical datum has not been independently identified; use these as relative heights. All pixels are finite, including areas where geological observations are masked. No DEM filling or analytical fitting was used.

Run `.venv/bin/python scripts/prepare_terrain.py` for an offline rebuild. Add `--download` to refresh the public USGS export; live service updates can change the source raster. `terrain_hillshade.png` is a QA visualization only. Terrain does not resolve the nonuniqueness of the subsurface history or the geological map's thin-unit sampling limits.
