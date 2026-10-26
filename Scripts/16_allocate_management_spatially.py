
from aquaculture_paths import project_path, external_path
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(str(project_path('meta/Upgrade/04_spatial_prioritization/scripts/analyze_spatial.py'))).resolve().parents[1]
META = Path(str(project_path('meta')))
FIRST = META / 'Upgrade/03_factorial_high_IPSL_SSP585_DOcap10'
IDS = ['Baseline_none', 'DO_only_high', 'N_only_high', 'pH_only_high', 'DO_N_high', 'DO_pH_high', 'N_pH_high', 'DO_N_pH_high']
LABELS = ['No action', 'DO', 'N', 'pH', 'DO + N', 'DO + pH', 'N + pH', 'DO + N + pH']
PERIODS = {'full': ('2025', '2100'), 'early': ('2025', '2060'), 'late': ('2061', '2100')}
COVERAGE = np.arange(101) / 100

def write_csv(frame, name):
    frame.to_csv(ROOT / 'tables' / name, index=False, encoding='utf-8-sig')

def curve(area, benefits, order):
    cumulative_area = np.r_[0, np.cumsum(area[order])]
    cumulative_benefit = np.r_[0, np.cumsum(benefits[order])]
    return np.interp(COVERAGE * area.sum(), cumulative_area, cumulative_benefit)

def allocation(area, order, share):
    remaining = share * area.sum() - np.r_[0, np.cumsum(area[order])[:-1]]
    fractions = np.clip(remaining / area[order], 0, 1)
    restored = np.zeros(len(area))
    restored[order] = fractions
    if not np.isclose((restored * area).sum(), share * area.sum(), rtol=1e-10, atol=1e-3):
        raise ValueError('Coverage constraint failed')
    return restored

def main():
    for name in ['data', 'tables', 'figures', 'docs', 'logs']:
        (ROOT / name).mkdir(exist_ok=True)
    print('1. Checking China-boundary pond inventory', flush=True)
    pond_path = META / 'Fig 5/Provincial_emission_intensity_pH_median/data/pond_level_GHG_flux_emission_pH_median.csv'
    boundary_path = pond_path.parent / 'china_province_boundary_aliyun_100000_full.geojson'
    ponds = pd.read_csv(pond_path, usecols=['Province', 'lon', 'lat', 'area_m2'])
    ponds.insert(0, 'pond_id', np.arange(1, len(ponds) + 1))
    boundary = gpd.read_file(boundary_path).to_crs(4326)
    invalid_boundary_count = int((~boundary.geometry.is_valid).sum())
    boundary.geometry = boundary.geometry.make_valid()
    inside = gpd.GeoSeries(gpd.points_from_xy(ponds.lon, ponds.lat), crs=4326).within(boundary.geometry.union_all())
    if not inside.all():
        raise ValueError(f'{(~inside).sum()} cached pond centroids are outside the boundary; reconcile before analysis')
    if not np.isfinite(ponds[['lon', 'lat', 'area_m2']]).all().all() or (ponds.area_m2 <= 0).any():
        raise ValueError('Invalid pond coordinates/areas')
    paths = [pond_path, boundary_path]
    total_files = [FIRST / f'Future_Run/Total_GHG/Total_CO2eq_China_IPSL-CM6A-LR_ssp585_{identifier}_OptimizedBounds.nc' for identifier in IDS]
    with xr.open_dataset(total_files[0]) as ds:
        grid_lon, grid_lat = np.meshgrid(ds.lon.values, ds.lat.values)
        valid = np.isfinite(ds.co2eq_flux.mean('time').values)
        coordinates = np.column_stack((grid_lon[valid], grid_lat[valid]))
        distance, neighbors = cKDTree(coordinates).query(ponds[['lon', 'lat']].to_numpy(), k=5)
        weights = 1 / (distance + 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        lat_reference = ds.lat.values.copy()
        lon_reference = ds.lon.values.copy()
    ponds['grid_id'] = neighbors[:, 0]
    area = ponds.area_m2.to_numpy()
    grid_area = np.bincount(neighbors.ravel(), weights=(weights * area[:, None]).ravel(), minlength=len(coordinates))
    def extract(path, variable, scale=1):
        paths.append(path)
        output = {}
        with xr.open_dataset(path) as ds:
            if not np.array_equal(ds.lat.values, lat_reference) or not np.array_equal(ds.lon.values, lon_reference):
                raise ValueError(f'Grid mismatch: {path}')
            flux = ds[variable].sel(time=slice('2025', '2100')).transpose('time', 'lat', 'lon')
            if flux.sizes['time'] != 912 or len(np.unique(flux.time.dt.strftime('%Y-%m').values)) != 912:
                raise ValueError(f'Incomplete months: {path}')
            if not np.array_equal(np.isfinite(flux.values).all(axis=0), valid):
                raise ValueError(f'Invalid mask/time-varying missing data: {path}')
            for period, (start, end) in PERIODS.items():
                subset = flux.sel(time=slice(start, end))
                cumulative = (subset * subset.time.dt.days_in_month * 24 * scale).sum('time').values[valid]
                emission = (cumulative[neighbors] * weights).sum(axis=1) * area / 1e15
                independent = (cumulative * grid_area).sum() / 1e15
                if not np.isclose(emission.sum(), independent, rtol=1e-10, atol=1e-8):
                    raise ValueError('Pond vs weighted-grid integration mismatch')
                output[period] = emission
        return output
    original = pd.read_csv(FIRST / 'tables/Factorial_summary_2025_2100.csv').set_index('Candidate_ID')
    reconcile = []
    for identifier, path in zip(IDS, total_files):
        print(f'2. Integrating {identifier}', flush=True)
        for period, values in extract(path, 'co2eq_flux').items():
            ponds[f'E_{period}_{identifier}'] = values
        current = ponds[f'E_full_{identifier}'].sum()
        previous = original.loc[identifier, 'Cumulative_Tg_CO2eq']
        reconcile.append({'Candidate_ID': identifier, 'Round1_Tg': previous, 'China_boundary_Tg': current,
                          'Difference_Tg': current - previous, 'Difference_percent': (current / previous - 1) * 100})
    write_csv(pd.DataFrame(reconcile), '01_boundary_reconciliation.csv')
    for period in PERIODS:
        for identifier in IDS:
            ponds[f'B_{period}_{identifier}'] = ponds[f'E_{period}_Baseline_none'] - ponds[f'E_{period}_{identifier}']
    for identifier in IDS:
        gas_sum = np.zeros(len(ponds))
        for gas, scale in [('CH4', 27.2), ('CO2', 1), ('N2O', .273)]:
            path = FIRST / f'Future_Run/{gas}_Management/{gas}_China_IPSL-CM6A-LR_ssp585_{identifier}_OptimizedBounds.nc'
            values = extract(path, f'{gas.lower()}_flux', scale)['full']
            ponds[f'E_{gas}_{identifier}'] = values
            gas_sum += values
        if not np.allclose(gas_sum, ponds[f'E_full_{identifier}'], rtol=2e-5, atol=1e-8):
            raise ValueError('Gas mass conversion or sum mismatch')
    ponds.to_csv(ROOT / 'data/pond_scenarios_China_boundary.csv.gz', index=False, compression='gzip')
    aggregations = {column: 'sum' for column in ponds if column.startswith(('E_', 'B_'))}
    aggregations['area_m2'] = 'sum'
    grid = ponds.groupby('grid_id').agg(aggregations).sort_index()
    grid['lon'] = coordinates[grid.index, 0]
    grid['lat'] = coordinates[grid.index, 1]
    grid['pond_count'] = ponds.groupby('grid_id').size()
    grid.reset_index().to_csv(ROOT / 'data/grid_scenarios.csv', index=False)
    write_csv(ponds.groupby('Province').agg(aggregations).reset_index(), '02_provincial_scenarios.csv')
    summary = pd.DataFrame({'Candidate_ID': IDS, 'Label': LABELS,
                            'Cumulative_Tg': [ponds[f'E_full_{identifier}'].sum() for identifier in IDS],
                            'Avoided_Tg': [ponds[f'B_full_{identifier}'].sum() for identifier in IDS]})
    summary['Reduction_percent'] = summary.Avoided_Tg / summary.Cumulative_Tg.iloc[0] * 100
    write_csv(summary, '03_boundary_factorial_summary.csv')
    print('3. Equal-area spatial allocation and temporal transfer', flush=True)
    areas = grid.area_m2.to_numpy()
    benefit = grid.B_full_DO_N_pH_high.to_numpy()
    baseline = grid.E_full_Baseline_none.to_numpy()
    orders = {'Baseline priority': np.argsort(-baseline / areas, kind='stable'),
              'Mitigation priority': np.argsort(-benefit / areas, kind='stable')}
    full_curves = {name: curve(areas, benefit, order) for name, order in orders.items()}
    rng = np.random.default_rng(42)
    draws = np.array([curve(areas, benefit, rng.permutation(len(grid))) for repeat in range(500)])
    curves = pd.DataFrame({'Coverage_percent': COVERAGE * 100, **full_curves,
                           'Random_mean': draws.mean(axis=0), 'Random_p025': np.quantile(draws, .025, axis=0), 'Random_p975': np.quantile(draws, .975, axis=0)})
    write_csv(curves, '04_coverage_curves_Tg.csv')
    if not np.allclose(draws[:, -1], benefit.sum()) or not all(np.isclose(values[-1], benefit.sum()) for values in full_curves.values()):
        raise ValueError('Full coverage does not match national avoided emissions')
    details = []
    provincial_allocations = []
    for share in [.1, .2, .3, .5]:
        for method, order in orders.items():
            fractions = allocation(areas, order, share)
            details.append({'Coverage_percent': share * 100, 'Method': method, 'Area_km2': (areas * fractions).sum() / 1e6,
                            'Avoided_Tg': (benefit * fractions).sum(), 'Share_of_full_implementation_percent': (benefit * fractions).sum() / benefit.sum() * 100})
            selected = ponds[['Province', 'grid_id', 'area_m2', 'B_full_DO_N_pH_high']].copy()
            selected['fraction'] = selected.grid_id.map(pd.Series(fractions, index=grid.index))
            selected['Managed_area_km2'] = selected.area_m2 * selected.fraction / 1e6
            selected['Avoided_Tg'] = selected.B_full_DO_N_pH_high * selected.fraction
            province = selected.groupby('Province').agg(Pond_area_m2=('area_m2', 'sum'), Managed_area_km2=('Managed_area_km2', 'sum'), Avoided_Tg=('Avoided_Tg', 'sum')).reset_index()
            province['Managed_percent'] = province.Managed_area_km2 * 1e6 / province.Pond_area_m2 * 100
            province['Method'] = method
            province['Coverage_percent'] = share * 100
            provincial_allocations.append(province)
    write_csv(pd.DataFrame(details), '05_selected_coverage_comparison.csv')
    provincial = pd.concat(provincial_allocations, ignore_index=True)
    write_csv(provincial, '06_provincial_allocation.csv')
    early_benefit = grid.B_early_DO_N_pH_high.to_numpy()
    late_benefit = grid.B_late_DO_N_pH_high.to_numpy()
    early_order = np.argsort(-early_benefit / areas, kind='stable')
    early_baseline_order = np.argsort(-grid.E_early_Baseline_none.to_numpy() / areas, kind='stable')
    late_order = np.argsort(-late_benefit / areas, kind='stable')
    transfer = pd.DataFrame({'Coverage_percent': COVERAGE * 100, 'Early_mitigation_ranking': curve(areas, late_benefit, early_order),
                             'Early_baseline_ranking': curve(areas, late_benefit, early_baseline_order), 'Late_oracle_ranking': curve(areas, late_benefit, late_order)})
    write_csv(transfer, '07_temporal_transfer_late_period_Tg.csv')
    print('4. Regional combination screening', flush=True)
    combo_rows = []
    benefits = grid[[f'B_full_{identifier}' for identifier in IDS]].to_numpy()
    sizes = np.array([0, 1, 1, 1, 2, 2, 2, 3])
    for threshold in [.9, .95]:
        for position, (grid_id, row) in enumerate(grid.iterrows()):
            full_benefit = benefits[position, -1]
            if full_benefit <= 0:
                selected_index = 0
                status = 'Full combination nonpositive; inspect separately'
            else:
                eligible = np.flatnonzero(benefits[position] >= threshold * full_benefit)
                fewest = eligible[sizes[eligible] == sizes[eligible].min()]
                selected_index = fewest[np.argmax(benefits[position, fewest])]
                status = 'Fewer measures; not a cost optimum'
            combo_rows.append({'grid_id': grid_id, 'Threshold': threshold, 'Selected_ID': IDS[selected_index], 'Label': LABELS[selected_index],
                               'Measure_count': int(sizes[selected_index]), 'Area_km2': row.area_m2 / 1e6,
                               'Full_avoided_Tg': full_benefit, 'Selected_avoided_Tg': benefits[position, selected_index], 'Status': status})
    combos = pd.DataFrame(combo_rows)
    write_csv(combos, '08_grid_combination_screen.csv')
    write_csv(combos.groupby(['Threshold', 'Label'], as_index=False).agg(Area_km2=('Area_km2', 'sum'), Selected_avoided_Tg=('Selected_avoided_Tg', 'sum')), '09_combination_area_summary.csv')
    for identifier in ['pH_only_high', 'DO_pH_high', 'DO_N_pH_high']:
        grid[f'CH4_B_{identifier}'] = grid.E_CH4_Baseline_none - grid[f'E_CH4_{identifier}']
    write_csv(grid.reset_index()[['grid_id', 'lon', 'lat', 'area_m2'] + [column for column in grid if column.startswith('CH4_B_')]], '10_CH4_tradeoffs.csv')
    print('5. PDF figures', flush=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout='constrained')
    axes[0].fill_between(curves.Coverage_percent, curves.Random_p025, curves.Random_p975, color='.85', label='Random 95% range')
    axes[0].plot(curves.Coverage_percent, curves.Random_mean, color='.4', linestyle=':', label='Random mean')
    for name, color, style in [('Baseline priority', '#4477AA', '--'), ('Mitigation priority', '#228833', '-')]:
        axes[0].plot(curves.Coverage_percent, curves[name], color=color, linestyle=style, label=name)
    for name, label, color, style in [('Early_baseline_ranking', 'Early baseline ranking', '#4477AA', '--'), ('Early_mitigation_ranking', 'Early mitigation ranking', '#228833', '-'), ('Late_oracle_ranking', 'Late-period optimum', '.4', ':')]:
        axes[1].plot(transfer.Coverage_percent, transfer[name], color=color, linestyle=style, label=label)
    for axis, title in zip(axes, ['(a) Allocation during 2025–2100', '(b) Early ranking evaluated in 2061–2100']):
        axis.set(xlabel='Managed pond area (%)', ylabel='Avoided emissions (Tg CO₂-eq)', xlim=(0, 100), title=title)
        axis.legend(frameon=False, fontsize=7)
    fig.savefig(ROOT / 'figures/Fig_spatial_coverage_and_transfer.pdf', bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), layout='constrained')
    for axis, method in zip(axes, orders):
        subset = provincial[(provincial.Method == method) & (provincial.Coverage_percent == 20)]
        mapped = boundary.merge(subset, left_on='name', right_on='Province', how='left').to_crs('+proj=aea +lat_1=25 +lat_2=47 +lon_0=105 +datum=WGS84')
        mapped.plot(column='Managed_percent', ax=axis, vmin=0, vmax=100, cmap='YlGnBu', legend=axis is axes[1], missing_kwds={'color': '#eeeeee'}, edgecolor='white', linewidth=.3)
        axis.set_title(method); axis.set_axis_off()
    fig.suptitle('Provincial managed pond area (%) at 20% national coverage', fontsize=10)
    fig.savefig(ROOT / 'figures/Fig_provincial_allocation_20percent.pdf', bbox_inches='tight'); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7.2, 3.5), layout='constrained')
    proportions = combos.groupby(['Label', 'Threshold']).Area_km2.sum().unstack(fill_value=0).reindex(LABELS, fill_value=0) / (areas.sum() / 1e6) * 100
    position = np.arange(len(LABELS))
    axis.barh(position-.18, proportions[.9], height=.35, color='#4477AA', label='90% of full-combination benefit')
    axis.barh(position+.18, proportions[.95], height=.35, color='#EEAA33', hatch='//', label='95% of full-combination benefit')
    axis.set_yticks(position, LABELS); axis.invert_yaxis(); axis.set_xlabel('Share of mapped pond area (%)'); axis.legend(frameon=False, fontsize=8)
    fig.savefig(ROOT / 'figures/Fig_fewer_measure_combinations.pdf', bbox_inches='tight'); plt.close(fig)
    manifest = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}
    (ROOT / 'docs/input_hashes.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    audit = {'pond_count': len(ponds), 'pond_area_km2': float(areas.sum()/1e6), 'allocation_grid_count': len(grid),
             'invalid_boundary_features_repaired_in_memory': invalid_boundary_count,
             'within_China_boundary': bool(inside.all()), 'monthly_steps': 912, 'random_repeats': 500,
             'baseline_Tg': float(summary.Cumulative_Tg.iloc[0]), 'full_combination_avoided_Tg': float(benefit.sum()),
             'negative_full_benefit_grid_count': int((benefit<0).sum()), 'negative_full_benefit_area_percent': float(areas[benefit<0].sum()/areas.sum()*100),
             'integration_and_full_coverage_checks': 'passed'}
    (ROOT / 'docs/validation.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (ROOT / 'docs/COMPLETED.txt').write_text('Completed boundary-constrained spatial analysis. See validation.json. No original projections overwritten.\n', encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)

if __name__ == '__main__':
    main()
