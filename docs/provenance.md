# Prior-work references

This Python implementation was written for this study after inspecting the following existing MATLAB code. It uses the same CR3BP sign conventions and the established separation between dynamics, measurements, reference frames and estimation. It is not a byte-for-byte translation or a reproduction of past numerical results.

## Gradient-Free-Optimization-for-Cislunar-SDA-2

Repository: https://github.com/scottblender/Gradient-Free-Optimization-for-Cislunar-SDA-2

- `src/orbitDynamics/cr3bp_dynamics.m`: Earth at −μ and Moon at 1−μ, normalized rotating dynamics and STM.
- `src/orbitDynamics/cr3bp_jacobian.m`: analytic Hessian/Coriolis blocks.
- `scripts/build_observer_orbit_catalog.m`: μ = 0.01215058560962404, LU = 384400 km, and SNRHL2 → southern NRHO/L2 family mapping.
- `src/measurements/measurement_model.m`: its AR case means angles plus **range**, not angular rates. This study introduces a distinct angles-plus-angular-rates measurement case.
- `src/targetGeneration/build_truth_gateway.m` and `target_case_config.m`: Gateway truth comes from the external TargetCaseDatabase.mat; no numerical Gateway state was available in those inspected source files.
- `README.md`: serif, large-font figure conventions and code organization.

## Lunar_Surface_Sensor_Network_Optimization

Repository: https://github.com/scottblender/Lunar_Surface_Sensor_Network_Optimization

- `src/+measurements/surfaceRaDecMeasurement.m`: topocentric inertial-axis angular measurements and east-positive site longitude.
- `src/+estimation/lunarSurfaceEkf.m`: six-state filtering, STM prediction, observation gating, wrapped angular innovations, Joseph updates.

The previous lunar two-body dynamics were not reused for L2 halo propagation. Terrain and DEM machinery are intentionally absent from this spherical-site baseline.

## Input provenance

The user supplied catalog row 241 / 6, family SNRHL2, T=1.8572 TU, stability=1.6875 and six four-decimal normalized state components. Identifier meanings and catalog epoch were not supplied. TU is an explicit approximate Earth–Moon normalization assumption; confirm against the original catalog before interpreting physical timing. No orbit database or private research results are bundled.
