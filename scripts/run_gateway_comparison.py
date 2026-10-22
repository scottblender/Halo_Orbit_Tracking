"""Run the one-observer-period perfect-versus-LSO-estimated comparison."""
import argparse
from pathlib import Path
from halo_tracking.study import load_config
from halo_tracking.gateway_comparison import run_gateway_comparison,save_comparison
from halo_tracking.plotting import geometry,observer_knowledge,gateway_comparison_figures,state_histories,export

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=None)
    parser.add_argument('--output',default='results/gateway_comparison')
    args=parser.parse_args()
    result=run_gateway_comparison(load_config(args.config))
    report=save_comparison(result,args.output)
    figures={'geometry':geometry(result),'observer':observer_knowledge(result),**gateway_comparison_figures(result),
             'observer_states':state_histories(result,'observer'),'gateway_states':state_histories(result,'gateway')}
    for name,figure in figures.items():export(figure,Path(args.output)/(name+'.png'))
    for name in ['perfect','lso_estimated']:
        print(name,report[name],flush=True)
