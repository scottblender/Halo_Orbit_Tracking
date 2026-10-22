"""Run uncontrolled departure with per-period trajectories and Gateway metrics."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from halo_tracking.study import load_config
from halo_tracking.departure import reference_departure, save_departure
from halo_tracking.gateway_comparison import run_gateway_comparison, save_comparison
from halo_tracking.plotting import departure_figures, period_trajectory_figures, gateway_comparison_figures, export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=None)
    parser.add_argument('--output',default='results/departure')
    parser.add_argument('--periods',type=int,default=20)
    parser.add_argument('--samples-per-period',type=int,default=500)
    parser.add_argument('--skip-gateway',action='store_true')
    args=parser.parse_args()
    cfg=load_config(args.config)
    output=Path(args.output)
    departure=reference_departure(cfg,args.periods,args.samples_per_period)
    save_departure(departure,cfg,output)
    for name,fig in {**departure_figures(departure,cfg),**period_trajectory_figures(departure,cfg)}.items():
        export(fig,output/(name+'.png'))
        plt.close(fig)
    print('Saved per-period departure figures:',output,flush=True)
    if not args.skip_gateway:
        result=run_gateway_comparison(cfg,periods=args.periods)
        report=save_comparison(result,output/'gateway_tracking')
        for name,fig in gateway_comparison_figures(result).items():
            export(fig,output/'gateway_tracking'/(name+'.png'))
            plt.close(fig)
        print('Saved Gateway period metrics:',len(report['per_period']),'rows',flush=True)


if __name__=='__main__':
    main()
