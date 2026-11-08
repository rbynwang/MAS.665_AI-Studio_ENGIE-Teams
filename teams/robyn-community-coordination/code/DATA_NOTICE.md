# Data notice

**Every load in `data_year_t1b/` is simulated.** No home or building here is a real meter, and no real person's data is here. The pack holds 8 homes and 3 buildings, on two stretches of days, 1 to 28 Jan and 1 Jun to 3 Jul 2026, every 15 minutes.

The Apache License in `LICENSE` covers the code, not this data. The series FlexGrid simulated are offered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Each input below keeps its own terms, and they travel with anything built on it.

## What each dataset is

| Dataset | Where in the pack | Real or synthetic | Source | Terms |
|---|---|---|---|---|
| Home loads, solar, EV sessions, indoor temperature, device schedules, house constants, owner limits | `homes/`, `devices/*.npy`, `hvac_params.json`, `owners.json`, `fleet_v2.json` | **Synthetic.** A simulator's output: none of these homes exists | homesim, from the [MassHack project](https://github.com/rbynwang/MassHack) (`src/homesim`), extended for FlexGrid | CC BY 4.0, with the credits below |
| Building loads, cooling, lighting | `buildings/`, `buildings.json` | **Synthetic.** Simulated profiles, not those sites' meters. Sizes come from public figures, and shapes from a fitted weather model | FlexGrid's model. Yearly totals sized from Boston's [BERDO](https://data.boston.gov/dataset/building-emissions-reduction-and-disclosure-ordinance) disclosures | CC BY 4.0, with the credits below |
| Measured temperature, `temp_c` | every home and building CSV | **Real**, NOAA's measurements at five Massachusetts airports, with short gaps filled | [NOAA GHCNh](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly) | US government data, listed as [CC0](https://catalog.data.gov/dataset/global-historical-climatology-network-hourly-ghcnh-version-1). Cite as below. NOAA makes no warranty and endorses nothing here |
| Sunlight, `ghi_wm2` | every home and building CSV | **Real to 31 May 2026**, HouseZero's sensor. After that, ECMWF's forecast scaled to HouseZero month by month | [HouseZero](https://doi.org/10.6084/m9.figshare.32790984.v1), ECMWF | CC BY 4.0, both |
| Weather forecasts as issued, `fc_temp_c`, `fc_cloud_pct`, `fc_sw_wm2` | every home and building CSV | **Real** forecasts, as issued two days ahead | ECMWF IFS 0.25°, through the [Open-Meteo](https://open-meteo.com/) Previous Runs API | CC BY 4.0, both |
| Wholesale prices, `price_usd_mwh`, `price_da_usd_mwh` | every home and building CSV | **A placeholder**, $50 per MWh at every step. Not a market price | None | Free to use. ISO New England's real prices are not here, because its terms give no licence to republish them |
| Event days, `grid_events.json` | `grid_events.json` | **Derived by rule.** The days with New England's highest evening load, with program-style terms. Not real program calls. The buyer is a stand-in: ENGIE, named in its mock label here and in the code, has no part in this project | Our rule, on ISO New England's measured load. No load figure is included | Part of this pack |
| Program rates, `devices/price_book.json` | `devices/price_book.json` | **Real** published rates, and our conversion to a price per kWh, marked as ours | The sources named in the file | Facts, with their sources |
| Device figures, `devices/catalog.json`, product names in `fleet_v2.json` | as named | **Real** figures from makers' published specifications, and assumptions | Makers' datasheets, not copied here | Facts. Product names belong to their owners, and none of them has any part in this project |

## How the simulated series were made

**Homes.** homesim runs a one-node thermal model of each house, driven by outdoor temperature and sun, with occupant schedules, appliances, EV charging and rooftop solar on top. Every home's solar shape comes from HouseZero's measured array, scaled by size. homesim's settings were tuned so that the average simulated home matches the day shape of National Grid's R-1 class average. Household equipment follows the US Energy Information Administration's 2024 Residential Energy Consumption Survey. No National Grid or survey value is in the pack. Settings that nothing measured are assumptions.

**Buildings.** Each building's yearly total is sized from BERDO, and the shape comes from a model fitted on National Grid's business class averages and measured Boston weather. The split between cooling and other uses comes from public reports. The model's output is here, and no National Grid value is. The labels say "simulated" because the sizes were chosen to resemble large Boston buildings, and these are not those buildings' meters.

## Credits

- **HouseZero.** Harvard CGBC at GSD (2026). HouseZero® one year dataset (June 2025 - May 2026). figshare. Dataset. https://doi.org/10.6084/m9.figshare.32790984.v1. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The values were averaged to the hour, resampled to 15 minutes, used as a simulator's input, and scaled. This is a transformed derivative and not an official Harvard release or endorsement.
- **ECMWF.** © 2026 European Centre for Medium-Range Weather Forecasts (ECMWF). Source: [www.ecmwf.int](https://www.ecmwf.int). This data is published under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. The data has been modified: hourly values at one point per town were resampled to 15 minutes, and the sunlight series was scaled. ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.
- **Open-Meteo.** Weather data by [Open-Meteo.com](https://open-meteo.com/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reference: Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649. It was fetched for non-commercial course work.
- **NOAA.** Menne, Matthew J.; Noone, Simon; Casey, Nancy W.; Dunn, Robert H.; McNeill, Shelley; Kantor, Diana; Thorne, Peter W.; Orcutt, Karen; Cunningham, Sam; Risavi, Nicholas. Global Historical Climatology Network-Hourly (GHCNh), stations KBOS, KBED, KLWM, KORH and KOWD. NOAA National Centers for Environmental Information. https://doi.org/10.25921/jp3d-3v19. Accessed 30 Oct 2026.
- **City of Boston.** BERDO disclosures, Analyze Boston, [ODC PDDL](https://opendatacommons.org/licenses/pddl/).
- **US Energy Information Administration.** 2024 Residential Energy Consumption Survey, public data.
- **National Grid.** Massachusetts Electric class average load shapes guided the calibration. National Grid has no part in this project, and none of its data is redistributed here.

Nothing here is endorsed by any of these organisations.
