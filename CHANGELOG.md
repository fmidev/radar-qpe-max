# Changelog

## Unreleased (next major)

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
