"""Forecast covariance at 5, 10 and 20 nominal periods with ongoing LSO data."""
import argparse
from pathlib import Path
from halo_tracking.study import load_config
from halo_tracking.horizons import forecast_horizons, save_horizon_forecast
from halo_tracking.plotting import horizon_comparison, export

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--periods', nargs='+', type=float, default=[1,5,10,20])
    parser.add_argument('--output', default='results/horizons')
    args = parser.parse_args()
    result = forecast_horizons(load_config(args.config), args.periods)
    save_horizon_forecast(result, args.output)
    fig = horizon_comparison(result)
    export(fig, Path(args.output)/'horizon_uncertainty.png')
    for row in result['horizon_rows']:
        print(row['periods'], row['measurement_case'],
              'position RSS sigma km:',row['position_rss_sigma_km'],
              'velocity RSS sigma m/s:',row['velocity_rss_sigma_m_s'], flush=True)
