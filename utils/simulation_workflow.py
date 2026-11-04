# Simulation study for bNMF and sfNMF (manuscript revision).
#
# For every experiment, setting and replicate:
#   simulate_*  creates a synthetic dataset (simulate.py simulate --npz)
#   fit_*       fits one method to it and evaluates against the truth (simulate.py run --data)
#   merge_*     collects the result tables per setting, per experiment and for the whole study
#
# Each experiment changes one thing from the base scenario. Settings that end up with the
# same dataset (e.g. depth 1e6 in the noise experiment is the base dataset) share the
# simulate and fit targets.
#
# Run from this directory, after `uv sync` in the repository root on the cluster:
#   gwf -f simulation_workflow.py status
#   gwf -f simulation_workflow.py run 'simulate_*'    # only create the synthetic data
#   gwf -f simulation_workflow.py run                 # everything

from gwf import Workflow
import re

fit_threads = 2

gwf = Workflow(
    defaults={
        'cores': fit_threads,
        'memory': '4g',
        'walltime': '11:00:00',
        'account': 'MutationAnalysis'})


def legalize_target_name(target_name):

    # First check if target name is legal.
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9._]*$', target_name):
        return target_name

    # Target name is not legal. Replace illegal characters by underscores.
    result = ''

    # First character is special.
    if re.match(r'[a-zA-Z_]', target_name[0]):
        result += target_name[0]
    else:
        result += '_'

    # Check remaining characters.
    for i in range(1, len(target_name)):
        if re.match(r'[a-zA-Z0-9._]', target_name[i]):
            result += target_name[i]
        else:
            result += '_'

    return result


def label(params):
    """File-name friendly description of a set of options, 'base' if there are none."""
    if not params:
        return 'base'
    return '_'.join(f'{key}-{value}'.replace(',', '+') for key, value in sorted(params.items()))


def cli_args(params):
    return ' '.join(f'--{key} {value}' for key, value in sorted(params.items()))


# Python from the project's uv environment, and the simulation script.
python = '../.venv/bin/python'
simulate = 'simulate.py'
study_dir = 'simulation_study'

n_replicates = 20
methods = ['nmf', 'plain', 'bnmf', 'sfnmf', 'oracle']

# The base scenario. All of these are passed explicitly, so this file (not the defaults
# in simulate.py) defines the base scenario.
base_data = {
    'n_batches': 3,
    'n_per_batch': '50',
    'n_test': 50,
    'case_fraction': '0.5',
    'n_signatures': 5,
    'tumor_periodicity': 0.3,
    'alpha': 0.5,
    'tumor_fraction': '2,3',
    'filter_family': 'sigmoid',
    'filter_strength': 1,
    'steepness': 1,
    'sample_filter_sd': 0,
    'depth': '1e6',
    'dm_concentration': 0}
base_fit = {
    'n_starts': 5,
    'test_chunk_sizes': '50'}
# Options that only change how the data are fitted; everything else changes the data.
fit_keys = {'n_starts', 'test_chunk_sizes', 'n_signatures_fit', 'volume_regularization', 'tol'}

# One factor at a time around the base scenario.
experiments = {
    'base': [{}],
    'noise': [{'depth': d} for d in ['1e4', '1e5', '1e6', '1e7', 'inf']]
             + [{'dm_concentration': c} for c in ['1e6', '1e5', '1e4']],
    'size': [{'n_per_batch': n} for n in ['5', '10', '25', '50', '100', '200']] + [{'n_per_batch': '100,25,10'}],
    'batches': [{'n_batches': k} for k in [2, 3, 5, 10]],
    'filter_family': [{'filter_family': f} for f in ['sigmoid', 'short_depletion', 'smooth', 'periodic', 'noisy']],
    'filter_strength': [{'filter_strength': s} for s in [0, 0.5, 1, 2, 3]],
    'steepness': [{'steepness': s} for s in [0.5, 1, 2, 4, 8]],
    'sample_heterogeneity': [{'sample_filter_sd': t} for t in [0, 0.05, 0.1, 0.2]],
    'signatures': [{'n_signatures': c} for c in [2, 5, 10, 20]],
    'signatures_fit': [{'n_signatures_fit': c} for c in [3, 4, 5, 6, 7]],
    'identifiability': [{'alpha': a} for a in [0.1, 0.3, 1, 3]] + [{'tumor_fraction': t} for t in ['2,8', '1,1']],
    'confounding': [{'case_fraction': cf, 'tumor_periodicity': p}
                    for cf in ['0.5', '0.2,0.5,0.8', '0.1,0.5,0.9'] for p in [0.3, 0.6]],
    'short_depletion': [{'filter_family': 'short_depletion', 'filter_strength': s, 'depth': d}
                        for s in [1, 2, 4] for d in ['1e5', '1e7']],
    'test_batch_size': [{'test_chunk_sizes': '50,20,10,5,2,1'}]}

simulated = set()
fitted = set()
experiment_files = []
for experiment, settings in experiments.items():
    setting_files = []
    for setting in settings:
        data_params = {**base_data, **{k: v for k, v in setting.items() if k not in fit_keys}}
        fit_params = {**base_fit, **{k: v for k, v in setting.items() if k in fit_keys}}
        data_label = label({k: v for k, v in data_params.items() if base_data.get(k) != v})
        fit_label = label({k: v for k, v in fit_params.items() if base_fit.get(k) != v})

        result_files = []
        for rep in range(1, n_replicates + 1):
            # Create the synthetic dataset.
            data_file = f'{study_dir}/data/{data_label}/rep{rep}.npz'
            if data_file not in simulated:
                simulated.add(data_file)
                gwf.target(
                    legalize_target_name(f'simulate_{data_label}_rep{rep}'),
                    inputs=[],
                    outputs=[data_file],
                    cores=1,
                    memory='2g',
                    walltime='00:59:00') << \
                    f'''
                    set -euo pipefail
                    mkdir -p $(dirname {data_file})
                    {python} {simulate} simulate --replicate {rep} {cli_args(data_params)} --npz {data_file}.tmp.npz
                    mv {data_file}.tmp.npz {data_file}
                    '''

            # Fit and evaluate each method.
            for method in methods:
                outfile = f'{study_dir}/fits/{data_label}/{fit_label}/rep{rep}_{method}.tsv'
                result_files.append(outfile)
                if outfile in fitted:
                    continue
                fitted.add(outfile)
                fit_name = f'fit_{method}_{data_label}' if fit_label == 'base' else f'fit_{method}_{data_label}_{fit_label}'
                gwf.target(
                    legalize_target_name(f'{fit_name}_rep{rep}'),
                    inputs=[data_file],
                    outputs=[outfile]) << \
                    f'''
                    set -euo pipefail
                    mkdir -p $(dirname {outfile})
                    export OMP_NUM_THREADS={fit_threads}
                    {python} {simulate} run --data {data_file} --methods {method} {cli_args(fit_params)} --out {outfile}.tmp.tsv
                    mv {outfile}.tmp.tsv {outfile}
                    '''

        # Results of one setting, labelled with the experiment and the setting.
        setting_label = label(setting)
        outfile = f'{study_dir}/experiments/{experiment}/{setting_label}.tsv'
        gwf.target(
            legalize_target_name(f'merge_{experiment}_{setting_label}'),
            inputs=result_files,
            outputs=[outfile],
            cores=1,
            memory='1g',
            walltime='00:59:00') << \
            f'''
            set -euo pipefail
            mkdir -p $(dirname {outfile})
            awk -v experiment={experiment} -v setting={setting_label} \\
                'BEGIN {{OFS = "\\t"}} FNR == 1 {{if (NR == 1) print "experiment", "setting", $0; next}} {{print experiment, setting, $0}}' \\
                {' '.join(result_files)} > {outfile}.tmp
            mv {outfile}.tmp {outfile}
            '''
        setting_files.append(outfile)

    # Results of one experiment.
    outfile = f'{study_dir}/experiments/{experiment}.tsv'
    gwf.target(
        f'merge_{experiment}',
        inputs=setting_files,
        outputs=[outfile],
        cores=1,
        memory='1g',
        walltime='00:59:00') << \
        f'''
        set -euo pipefail
        mkdir -p $(dirname {outfile})
        awk 'FNR > 1 || NR == 1' {' '.join(setting_files)} > {outfile}.tmp
        mv {outfile}.tmp {outfile}
        '''
    experiment_files.append(outfile)

# All results in one table.
outfile = f'{study_dir}/simulation_results.tsv'
gwf.target(
    'merge_all',
    inputs=experiment_files,
    outputs=[outfile],
    cores=1,
    memory='1g',
    walltime='00:59:00') << \
    f'''
    set -euo pipefail
    mkdir -p $(dirname {outfile})
    awk 'FNR > 1 || NR == 1' {' '.join(experiment_files)} > {outfile}.tmp
    mv {outfile}.tmp {outfile}
    '''
