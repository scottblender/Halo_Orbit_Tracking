"""Run and save the baseline without opening Jupyter."""
import argparse
from pathlib import Path
from halo_tracking.study import load_config,run_study,save_study
from halo_tracking.plotting import geometry,kinematics,measurement_history,estimation_errors,uncertainty_comparison,export

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default=None)
    parser.add_argument('--covariance-only',action='store_true'); parser.add_argument('--output',default='results/baseline')
    args=parser.parse_args(); result=run_study(load_config(args.config),args.covariance_only)
    save_study(result,args.output)
    figures={'geometry':geometry(result),'kinematics':kinematics(result),'measurements':measurement_history(result),
        'angles_errors':estimation_errors(result),'angles_rates_errors':estimation_errors(result,'angles_rates'),
        'uncertainty':uncertainty_comparison(result)}
    for name,fig in figures.items():export(fig,Path(args.output)/'figures'/(name+'.png'))
    print(result['diagnostics']); print('Saved to',args.output)
if __name__=='__main__':main()
