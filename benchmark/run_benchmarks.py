#!/usr/bin/env python3

# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# This script allows to gather the performance results reported by MIOpenDriver for (some of) the
# MIOpen layers and output a JSON with the relevant metrics.
#
# By default all the benchmarks will be run according to the layers, types and arguments
# from the shapes present in the 'include' entry of the benchmarking matrix (JSON file in this folder).
# It is also possible to run the benchmarks for the whole matrix, as well as for a subset of the
# layers. In the latter case, the layers in the subset and the type specified for them must be present
# in the benchmarking matrix. Otherwise the script will fail to run the benchmarks.
#
# The script will produce one JSON file for each layer(+type) benchmarked. The output files are named
# following the convention 'benchmark_<layer><type>_<arch>.json', where <arch> is the architecture
# version (gfxXXXX) of the device used for the execution of the benchmarks.
#
# Example of use:
#
# // Run benchmarks for all shapes
# python3 run_benchmarks.py
#
# // Run benchmarks for all shapes with custom-built MIOpenDriver
# python3 run_benchmarks.py --miopen-cmd "../build/bin/MIOpenDriver"
#
# // Run benchmarks for all shapes for one layer and type (convfp16)
# python3 run_benchmarks.py --layers conv --types fp16
#
# // Run benchmarks for all shapes for several layers and types (convfp16, convint8, and bnormfp16)
# python3 run_benchmarks.py --layers conv bnorm --types fp16,int8 fp16
#
# // Run benchmarks for whole matrix
# python3 run_benchmarks.py --full-bench
#

import hashlib
import argparse
import os
import sys
import json
import logging
import rich.logging
import rich.progress
import subprocess
import pandas as pd
from typing import Any, Dict, List
from itertools import product
import re

# Read benchmarking matrix (JSON file)
def read_json(json_path: str):
    try:
        with open(json_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        log.error(f'Could not read JSON file {json_path}: {e}')
        sys.exit(1)

# Run benchmarks
def parse_output(output: str) -> List[Dict[str, Any]]:
    '''
    Parse MIOpenDriver mem stats and kernel time report:
        stdstats: <layer>, <readbytes>, <writebytes>, <mem_bw>, <time>
        GPU Kernel Time <layer> Elapsed: <time> ms
    '''
    results = []
    solution_regex = r'Solution:\s*(?P<solution_id>\d+)/(?P<solution_name>\S+)'
    stats_regex = r'stdstats:\s*(?P<layer>\S+),\s*(?P<read_bytes>\d+),\s*(?P<write_bytes>\d+),\s*(?P<mem_bw_gbs>[\d\.]+),\s*(?P<time_ms>[\d\.]+)'

    solution_match = re.search(solution_regex, output)
    solution = f"{solution_match.group('solution_id')}/{solution_match.group('solution_name')}"

    # Some layers report more than one stdstats (e.g. fwd, bwdd, bwdw)
    stats_matches = re.finditer(stats_regex, output)
    for match in stats_matches:
        layer = match.group('layer')
        read_bytes = int(match.group('read_bytes'))
        write_bytes = int(match.group('write_bytes'))
        mem_bw_gbs = float(match.group('mem_bw_gbs'))
        time_ms = float(match.group('time_ms'))
        results.append({
            'layer': f'{layer}{type}',
            'read_bytes': read_bytes,
            'write_bytes': write_bytes,
            'mem_bw_gbs': mem_bw_gbs,
            'solution': solution,
            'time_ms': time_ms
        })
    return results

def convert_std_to_relative(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Convert DataFrame's std from absolute value to relative with respect to mean.
    '''
    for col in df.columns:
        if col.endswith('_std'):
            std_col = col
            mean_col = std_col.replace('_std', '_mean')
            if mean_col in df:
                df[std_col] = (df[std_col] / df[mean_col]) * 100
                df.loc[df[mean_col] == 0, std_col] = 0

    return df

def export_results(out_dir: str, out_file:str, results: list) -> None:
    '''
    Write results to JSON file.
    '''
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

def benchmark_layer(layer: str, type: str, iters: int, out_dir: str, miopen_cmd: str, arch: str, bandwidth_gbs: float, params_list: list, incremental: bool) -> None:
    '''
    The core benchmarking procedure:
    1) Gather results from MIOpenDriver for a single layer and type and all combinations of arguments' values.
    2) Compute metrics (arithmetic mean, standard deviation).
    3) Export results to JSON.
    '''
    result_dir = os.path.join(out_dir, 'benchmark_results')
    result_file = os.path.join(result_dir, f'benchmark_{layer}{f"_{type}" if type else ""}_{arch}.json')

    # Load existing results if run is incremental and results file exists
    existing_results = []
    existing_commands = {}
    if incremental and os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                existing_results = json.load(f)
                existing_commands = {r['command'] for r in existing_results}
        except:
            pass
    results = list(existing_results)

    for params in params_list:
        # Get command for each combination of args
        driver_command = [miopen_cmd, f'{layer}{type}']
        args_dict = {}
        for key, value in params.items():
            # If time set to 0, set to 1
            if key == 'time' and value == 0:
                value = 1
            driver_command.extend([f'--{key}', str(value)])
            args_dict[key] = value
        driver_command_str = ' '.join(driver_command)

        # Manually add --time 1 if not in benchmarking matrix
        if 'time' not in args_dict:
            driver_command.extend([f'--time', str(1)])
            args_dict['time'] = 1
            log.info(f'Adding \'--time 1\' to command: {driver_command_str}')
            driver_command_str = ' '.join(driver_command)

        # Skip if already exists
        if incremental and driver_command_str in existing_commands:
            log.info(f'Skipping existing result for command: {driver_command_str}')
            continue

        try:
            # Get results for 'iters' runs
            log.info(f'Running {driver_command_str} for {iters} iters')
            runs_results = []
            for _ in range(iters):
                run_result = subprocess.run(driver_command, capture_output=True, text=True, check=True)
                parsed_run_result = parse_output(run_result.stdout)
                runs_results.extend(parsed_run_result)
                if args.log:
                    params_hash = hashlib.md5(driver_command_str.encode()).hexdigest()[:8]
                    out_file = f"{logs_out_dir}/log_{layer}{type}_{arch}_{params_hash}"
                    with open(out_file, 'w') as f:
                        print(f"{run_result.stderr}\n{driver_command_str}", file=f)
                    break
            # Compute metrics (arithmetic mean, standard deviation)
            df = pd.DataFrame(runs_results)
            averaged_result = df.groupby(['layer', 'solution']).agg(['mean', 'std'])
            # Flatten metrics, stddev is reported in % with respect to the mean
            averaged_result.columns = ['_'.join(col) for col in averaged_result.columns]
            averaged_result = averaged_result.reset_index()
            convert_std_to_relative(averaged_result)
            # Store averaged results
            results.append({
                'command': driver_command_str,
                'args': args_dict,
                'device_arch': arch,
                'bandwidth_gbs': bandwidth_gbs,
                'results': averaged_result.to_dict(orient='records')})
        except subprocess.CalledProcessError as e:
            if "Unsupported layout" in e.stderr:
                log.warning(f'Ignoring \'Unsupported layout\' error for {driver_command_str}')
                continue
            log.error(f'Error benchmarking {driver_command_str}: {e}')
            sys.exit(1)

    # Export results to JSON
    export_results(result_dir, result_file, results)

def benchmark_layer_full(layer: str, type: str, iters: int, out_dir: str, miopen_cmd: str, arch:str, bandwidth_gbs: float, incremental: bool) -> None:
    '''
    Benchmark MIOpen layer for all the benchmarking matrix combinations.
    '''
    type_config = bench_matrix[layer][type]
    args_keys = list(type_config['args'].keys())
    args_values = list(type_config['args'].values())
    params_list = [dict(zip(args_keys, values)) for values in product(*args_values)]
    benchmark_layer(layer, type, iters, out_dir, miopen_cmd, arch, bandwidth_gbs, params_list, incremental)

def benchmark_layer_shapes(layer: str, type: str, iters: int, out_dir: str, miopen_cmd: str, arch:str, bandwidth_gbs: float, incremental: bool) -> None:
    '''
    Benchmark MIOpen layer for all the shapes.
    '''
    type_config = bench_matrix[layer][type]
    params_list = type_config.get('include', [])
    benchmark_layer(layer, type, iters, out_dir, miopen_cmd, arch, bandwidth_gbs, params_list, incremental)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_benchmarks',
        description='Benchmark MIOpen layers',
    )

    parser.add_argument('-a', '--arch', default='gfx942', help='Architecture to target, default to gfx942')
    parser.add_argument('-bm', '--bench-matrix',
                        default=f'{os.path.dirname(os.path.realpath(__file__))}/bench_matrix.json',
                        help='Path to benchmarking matrix JSON file, default to bench_matrix.json from script\'s folder')
    parser.add_argument('-f', '--full-bench', action='store_true', help='Run full benchmarking matrix')
    parser.add_argument('--incremental', action='store_true', help='Do not re-run benchmarks with existing results file')
    parser.add_argument('-l', '--layers', nargs='+', help='Space-separated list of layers to benchmark')
    parser.add_argument('--log', action='store_true', help='Run in tuning mode by only storing logs and not gathering performance metrics')
    parser.add_argument('-ls', '--list', action='store_true', help='List available layers')
    parser.add_argument(
        '-o', '--out_dir', default=os.path.dirname(os.path.realpath(__file__)), help='Output directory, default to same as script\'s')
    parser.add_argument('-m', '--miopen-cmd', default='MIOpenDriver', help='MIOpen command, default to system installation')
    parser.add_argument('-t', '--types', nargs='+', help='Space-separated list of comma-separated lists of types for each layer')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    parser.add_argument('-b','--bandwidth-gbs', type=float, default=800.0, help='Max theoretical bandwidth of the GPU in GB/s, default to 800GB/s')
    parser.add_argument('-i', '--iters', type=int, default=2, help='Number of runs of each benchmark, default to 2')

    args = parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    logging.basicConfig(format='%(message)s', handlers=[rich.logging.RichHandler(rich_tracebacks=True, markup=True)], level=log_level)
    log = logging.getLogger('rich')

    # Get benchmarking matrix
    bench_matrix = read_json(args.bench_matrix)

    # Get layers and types and check correctness
    if not args.layers:
        args.layers = list(bench_matrix.keys())
    if not args.types:
        args.types = [','.join(bench_matrix[layer].keys()) for layer in args.layers]
    if len(args.layers) != len(args.types):
        parser.error('There must be at least one type specified for each layer')

    if args.iters < 2:
        parser.error('The number of iterations mush be at least 2')

    if args.list:
        for layer in args.layers:
            print(layer)
        quit()
        
    if args.log:
        logs_out_dir=f'{os.path.dirname(os.path.realpath(__file__))}/logs'
        os.makedirs(logs_out_dir, exist_ok=True)
        log.info(f'Running in log mode. Logs stored in {logs_out_dir}')

    for layer, types_list in zip(args.layers, args.types):
        types = types_list.split(',')
        for type in types:
            log.info(f'Benchmarking {layer}{type} for {args.arch}')
            if args.full_bench:
                benchmark_layer_full(layer=layer, type=type, iters=int(args.iters), out_dir=args.out_dir, miopen_cmd=args.miopen_cmd, arch=args.arch, bandwidth_gbs=args.bandwidth_gbs, incremental=args.incremental)
            else:
                benchmark_layer_shapes(layer=layer, type=type, iters=int(args.iters), out_dir=args.out_dir, miopen_cmd=args.miopen_cmd, arch=args.arch, bandwidth_gbs=args.bandwidth_gbs, incremental=args.incremental)
    log.info(f'Finished benchmarking for {args.arch}')
