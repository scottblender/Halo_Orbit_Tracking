"""Execute notebooks independently; allow a socket-free process mode."""
import argparse
import base64
import io
import os
from pathlib import Path
import subprocess
import sys
import nbformat

root=Path(__file__).resolve().parents[1]

def execute_in_process(path):
    """Execute one notebook in this fresh process with IPython rich output."""
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from matplotlib_inline.backend_inline import configure_inline_support
    shell=InteractiveShell.instance()
    configure_inline_support(shell,'inline')
    import matplotlib.pyplot as plt
    notebook=nbformat.read(path,as_version=4)
    count=0
    for cell in notebook.cells:
        if cell.cell_type!='code':
            continue
        count+=1
        print(f"Executing {path.name}, code cell {count}",flush=True)
        with capture_output() as captured:
            execution=shell.run_cell(cell.source,store_history=True)
        if execution.error_before_exec or execution.error_in_exec:
            raise RuntimeError(f'{path.name}, cell {count}: {execution.error_before_exec or execution.error_in_exec}')
        cell.execution_count=count
        outputs=[]
        if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
        for number in plt.get_fignums():
            fig=plt.figure(number)
            buffer=io.BytesIO()
            fig.savefig(buffer,format='png',dpi=110,bbox_inches='tight')
            outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode('ascii')},metadata={}))
            plt.close(fig)
        cell.outputs=outputs
    nbformat.write(notebook,path)
    print('Executed in isolated process:',path.name,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--in-process',action='store_true',help='Use isolated IPython processes when kernel sockets are unavailable')
    parser.add_argument('--one',type=Path,help="Execute just this notebook (default: Notebook 01 only)")
    args=parser.parse_args()
    if args.one:
        execute_in_process(args.one)
    else:
        paths=[root/'notebooks'/'01_narx_results.ipynb']
        for path in paths:
            if args.in_process:
                subprocess.run([sys.executable,__file__,'--one',str(path)],cwd=root,check=True)
            else:
                from nbclient import NotebookClient
                notebook=nbformat.read(path,as_version=4)
                NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
                nbformat.write(notebook,path)
                print('Executed in fresh kernel:',path.name,flush=True)
