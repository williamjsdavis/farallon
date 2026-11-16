# Third-party notices

The MIT License in [LICENSE](LICENSE) covers the original code in this repository. The material listed below was not written by the repository author and remains under its own terms. Listing it here grants no additional rights.

## 1. Sheep Mountain geological map and files derived from it

**Source:** Fiore Allwardt, P., Bellahsen, N., and Pollard, D. D., 2007, Curvature and fracturing based on global positioning system data collected at Sheep Mountain anticline, Wyoming: *Geosphere*, v. 3, no. 6, p. 408–421, Figure 1 (printed page 409). DOI: [10.1130/GES00088.1](https://doi.org/10.1130/GES00088.1). The figure's caption credits the map "from Bellahsen et al., 2006a; after Rioux, 1994". Full citations are in [papers/REFERENCES.md](papers/REFERENCES.md).

**Licence:** None known; rights held by the original publisher. The article is marked "© 2007 Geological Society of America" and "For permission to copy, contact editing@geosociety.org". Bellahsen et al. (2006a), published in the *Journal of Structural Geology*, may also hold rights in the drawing. These files are **not** covered by the MIT License.

The WEBP copy matches Figure 1. Its caption is the figure's caption, and its legend colours agree with a render of the article's page 409 to within 1–2 RGB values (`data/palette_audit.json`). It was downloaded with a web browser on 8 September 2026. The page it was downloaded from was not recorded.

| File | Relationship to the figure |
| --- | --- |
| `assets/Geological-map-of-Sheep-Mountain-anticline-The-Bighorn-River-dissects-the-fold.webp` | The figure as published (697 × 671 px WEBP; SHA-256 `df0054eeb1823ee849edc29020d6a75e801d25fff6f0225fc1278674ae6bdbeb`) |
| `data/target_source.png` | Unmodified crop of the figure (x 20–286, y 50–250) |
| `data/target.png` | Unit colours classified from the crop |
| `data/target_mask.png` | Observation mask computed from the crop |
| `data/target.npz` | Unit labels, mask, exclusion reasons and coordinates computed from the crop |
| `data/target.json` | Metadata for the above, including legend colours and tick positions read from the figure |

These files also show or contain material computed from the figure:

- `assets/*.gif`: screen recordings of the app. The "Observed" panel shows the source crop with its original linework.
- `data/runs/*.json`: each recording embeds a mismatch image computed against the derived labels, and the prediction clipped to the derived observation mask.
- `data/palette_audit.json`: legend colour measurements sampled from the figure and from a render of the article's page 409.

## 2. shadcn/ui components

**Files:** the 60 component files in `web/components/ui/`, plus `web/hooks/use-mobile.ts` and `web/lib/utils.ts`. The shadcn CLI generated these (style `base-nova`; see `web/components.json`).

**Source:** shadcn/ui, <https://ui.shadcn.com>, <https://github.com/shadcn-ui/ui>

**Licence:** MIT

```text
MIT License

Copyright (c) 2023 shadcn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. USGS 3DEP terrain data

**Files:**

| File | Contents |
| --- | --- |
| `data/terrain_source.tif` | Elevation raster exported from the service |
| `data/terrain.npz` | Heights resampled from that export |
| `data/terrain_hillshade.png` | QA hillshade rendered from the export |
| `data/terrain_service.json` | Service metadata returned by USGS |
| `data/terrain_export.json` | Export response from USGS |

`data/terrain.json` records the export request, raster hash and registration details. See `data/TERRAIN.md`.

**Source:** U.S. Geological Survey, 3D Elevation Program (3DEP) Bare Earth DEM Dynamic Image Service, <https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer>, retrieved 8 September 2026. Service credit: "USGS National Map 3D Elevation Program (3DEP). August 25, 2026."

**Licence:** Public domain (U.S. Government work), with no use restrictions. USGS states that "All 3DEP products are available, free of charge and without use restrictions": <https://www.usgs.gov/3d-elevation-program/about-3dep-products-services>

## 4. Cited papers (referenced, not bundled)

Copies of these papers are no longer included in this repository. Citations, DOIs and links are in [papers/REFERENCES.md](papers/REFERENCES.md).

| Paper | Rights | Used for |
| --- | --- | --- |
| Fiore Allwardt, Bellahsen and Pollard (2007), *Geosphere* 3(6), 408–421, [doi:10.1130/GES00088.1](https://doi.org/10.1130/GES00088.1) | © 2007 Geological Society of America | Source of the geological map (section 1) |
| Parquer, de Kemp, Brodaric and Hillier (2025), *Geoscientific Model Development* 18, 71–100, [doi:10.5194/gmd-18-71-2025](https://doi.org/10.5194/gmd-18-71-2025) | CC BY 4.0, © Author(s) 2025 | Design research (`RESEARCH.md`) |

## 5. Direct dependencies

These packages are installed by `uv` and `npm`; none is vendored in this repository. Each licence comes from the package's own metadata at the version locked in `uv.lock` or `web/package-lock.json`.

### Python (`pyproject.toml`)

| Package | Constraint | Locked | Licence |
| --- | --- | --- | --- |
| numpy | >=2.2,<2.6 | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| numba | >=0.63,<0.68 | 0.67.0 | BSD-2-Clause |
| scipy | >=1.15,<1.19 | 1.18.1 | BSD-3-Clause (wheels also bundle components under their own licences) |
| pillow | >=11,<13 | 12.3.0 | MIT-CMU |
| fastapi | >=0.115,<1 | 0.141.1 | MIT |
| uvicorn | >=0.34,<1 | 0.52.4 | BSD-3-Clause |
| openai | >=2.54,<3 | 2.54.0 | Apache-2.0 |
| python-dotenv | >=1,<2 | 1.2.3 | BSD-3-Clause |
| pytest (dev group) | >=8,<10 | 9.1.1 | MIT |

### npm (`web/package.json`)

| Package | Constraint | Locked | Licence |
| --- | --- | --- | --- |
| @base-ui/react | 1.7.0 | 1.7.0 | MIT |
| @shadcn/react | 0.3.0 | 0.3.0 | MIT |
| @types/three | ^0.185.4 | 0.185.4 | MIT |
| class-variance-authority | 0.7.1 | 0.7.1 | Apache-2.0 |
| clsx | 2.1.1 | 2.1.1 | MIT |
| cmdk | 1.1.1 | 1.1.1 | MIT |
| date-fns | 4.1.0 | 4.1.0 | MIT |
| embla-carousel-react | 8.5.2 | 8.5.2 | MIT |
| input-otp | 1.4.2 | 1.4.2 | MIT |
| lucide-react | 1.31.0 | 1.31.0 | ISC |
| react | 19.2.6 | 19.2.6 | MIT |
| react-day-picker | 9.8.1 | 9.8.1 | MIT |
| react-dom | 19.2.6 | 19.2.6 | MIT |
| react-resizable-panels | 4.5.8 | 4.5.8 | MIT |
| react-server-dom-webpack | 19.2.6 | 19.2.6 | MIT |
| recharts | 3.8.0 | 3.8.0 | MIT |
| shadcn | 4.18.0 | 4.18.0 | MIT |
| tailwind-merge | 3.6.0 | 3.6.0 | MIT |
| three | ^0.186.0 | 0.186.0 | MIT |
| tw-animate-css | 1.4.0 | 1.4.0 | MIT |
| vinext | 1.0.0-beta.5 | 1.0.0-beta.5 | MIT |

Development dependencies:

| Package | Constraint | Locked | Licence |
| --- | --- | --- | --- |
| @cloudflare/vite-plugin | 1.37.1 | 1.37.1 | MIT |
| @cloudflare/workers-types | 4.20260515.1 | 4.20260515.1 | MIT OR Apache-2.0 |
| @openai/sites-vite-plugin | 0.2.0 | 0.2.0 | MIT |
| @tailwindcss/postcss | 4.2.1 | 4.2.1 | MIT |
| @types/node | 22.19.19 | 22.19.19 | MIT |
| @types/react | 19.2.14 | 19.2.14 | MIT |
| @types/react-dom | 19.2.3 | 19.2.3 | MIT |
| @vitejs/plugin-react | 6.0.2 | 6.0.2 | MIT |
| @vitejs/plugin-rsc | 0.5.26 | 0.5.26 | MIT |
| oxfmt | 0.61.0 | 0.61.0 | MIT |
| oxlint | 1.76.0 | 1.76.0 | MIT |
| oxlint-tsgolint | 7.0.2001 | 7.0.2001 | MIT |
| tailwindcss | 4.2.1 | 4.2.1 | MIT |
| typescript | 5.9.3 | 5.9.3 | Apache-2.0 |
| vite | 8.0.13 | 8.0.13 | MIT |
| wrangler | 4.92.0 | 4.92.0 | MIT OR Apache-2.0 |
