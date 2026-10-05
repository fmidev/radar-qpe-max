# Changelog

## 3.0.0

> [!IMPORTANT]
> **Grid geometry changed: existing netCDF caches are incompatible and must be
> deleted when deploying this version.**

### Changed (breaking)

- Per-site grids have a pixel size of exactly `resolution` (previously
  `size*resolution/(size-1)`, e.g. 250.1221 m instead of 250 m).
- Per-site pixel edges lie on a fixed lattice of multiples of `resolution` in
  EPSG:3067, centred on the lattice node nearest to the radar. Grids of
  different radars, and the finrad bounds `[-208000, 6390000, 1072000, 7926000]`,
  line up without resampling. `size` must be even.
- All GeoTIFFs are written north-up (negative pixel height).
- `composite_max` and `composite_max_with_time` use the finest input resolution
  when `resolution` is not given, as documented (previously the first input's).
- CLI `-r/--resolution` is parsed as an integer.
- **Breaking:** daily max GeoTIFFs are named following the FMI radar GeoTIFF
  convention, e.g. `202610050000_fikor_max_24h_acrr_finrad250_raw.tif`
  (previously `fikor20261004max1 d1024px250m.tif`). The timestamp is the end of
  the UTC day, the window is given in whole hours and `size` is no longer part
  of the name. New helpers `max_tif_name` and `max_tif_glob`.
- `write_max_tifs` takes `dbz_field` instead of `corr` and no longer takes
  `size`; CLI `winmax` now marks attenuation corrected (`-z DBZHC`) output as
  `rawac`. Windows that are not whole hours are rejected.
