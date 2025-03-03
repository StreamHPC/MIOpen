#!/usr/bin/env python3

# This script allows to compare performance results gathered with the run_benchmarks.py script.
#
# The script will output a table for each pair of (base, new) JSON files with the parameters' values
# and the difference between the specified performance metrics (memory bandwidth or execution time)
# for each combination benchmarked. The percentage of difference is printed:
# * in green if the base metrics are equal or worse than the new ones,
# * in yellow if the base metrics are at most 1% better than the new ones or if the standard deviation
#   of the base benchmark' results is geq than the difference in performance,
# * and in red otherwise (regression in performance is significant).
#
# Example of use:
#
# // Compare base benchmarks with new ones
# python3 compare_benchmarks.py --time -b ./base_benchmark_results -c ./new_benchmark_results
#
# // Compare a single base benchmark with the new one
# python3 compare_benchmarks.py --memory ./base_benchmark_results/benchmark_bnorm.json ./new_benchmark_results/benchmark_bnorm.json
#

import argparse
import json
import sys
import os.path
import pandas as pd
from functools import reduce
from tabulate import tabulate
from colorama import Fore, Style
from tabulate import tabulate

parser = argparse.ArgumentParser(
    prog='compare_benchmarks',
    description='Compare MIOpen layers benchmark results')
parser.add_argument('files', nargs='*', type=str, help='JSON results')
parser.add_argument('-b', '--baseline-dir', help='Path to directory with baseline JSON files')
parser.add_argument('-c', '--compare-dir', help='Path to directory with JSON files to be compared against baseline')
parser.add_argument('-m', '--memory', action='store_true', default=False, help='Compare memory bandwidth')
parser.add_argument('-t', '--time', action='store_true', default=False, help='Compare execution time')

pd.set_option('display.colheader_justify', 'left')

args = parser.parse_args()

if not (args.memory or args.time):
    print('at least --memory or --time needs to be passed', file=sys.stderr)
    sys.exit(1)

if args.baseline_dir and args.compare_dir:
    baseline_files = sorted([os.path.join(args.baseline_dir, f) for f in os.listdir(args.baseline_dir) if f.endswith('.json')])
    compare_files = sorted([os.path.join(args.compare_dir, f) for f in os.listdir(args.compare_dir) if f.endswith('.json')])
    if(len(baseline_files) == 0 or len(compare_files) == 0):
        print('baseline and comparation folders must have at least one file each', file=sys.stderr)
        sys.exit(1)
    if(len(baseline_files) != len(compare_files)):
        print('baseline and comparation folders must have the same amount of files', file=sys.stderr)
        sys.exit(1)
    nfiles = len(baseline_files)
elif len(args.files) > 0:
    if len(args.files) < 2:
        print('at least one input baseline and comparation files should be passed', file=sys.stderr)
        sys.exit(1)
    if len(args.files) % 2:
        print('the same amount of input baseline and comparation files should be passed', file=sys.stderr)
        sys.exit(1)
    nfiles = len(args.files) // 2
    baseline_files = args.files[:nfiles]
    compare_files = args.files[nfiles:]

diff_label = 'diff (%)'
std_label = 'stddev (%)'

def pick_color(diff: float, stddev: float):
    if diff >= 0:
        color = Fore.GREEN
    elif -1 < diff < 0 or -stddev < diff:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    return color

for baseline_file, compare_file in zip(baseline_files, compare_files):
    with open(baseline_file, 'r') as f:
        bdata = json.load(f)
    with open(compare_file, 'r') as f:
        cdata = json.load(f)

    entries = []
    for bresults, cresults in zip(bdata,cdata):
        keys = bresults['args'].keys()
        for bentry, centry in zip(bresults['results'], cresults['results']):
            new_entry = {'layer': str(bentry['layer'])}
            new_entry.update({k: str(bresults['args'][k]) for k in keys})
            if args.memory and bentry['mem_bw_gbs_mean']:
                diff = (centry['mem_bw_gbs_mean'] - bentry['mem_bw_gbs_mean']) / bentry['mem_bw_gbs_mean'] * 100 if bentry['mem_bw_gbs_mean'] else 0
                stddev = bentry['mem_bw_gbs_std']
                color = pick_color(diff, stddev)
                new_entry[f'mem_{diff_label}'] = f'{color}{diff}{Style.RESET_ALL}'
                new_entry[f'mem_{std_label}'] = stddev
            if args.time and bentry['time_ms_mean']:
                diff = (centry['time_ms_mean'] - bentry['time_ms_mean']) / bentry['time_ms_mean'] * 100 if bentry['time_ms_mean'] else 0
                stddev = bentry['time_ms_std']
                color = pick_color(diff, stddev)
                new_entry[f'time_{diff_label}'] = f'{color}{diff}{Style.RESET_ALL}'
                new_entry[f'time_{std_label}'] = stddev
            entries.append(new_entry)

    columns = [('key', 'layer')]
    columns.extend([('key', k) for k in keys])
    columns.append(('compare', diff_label))
    columns.append(('compare', std_label))
    df = pd.DataFrame.from_dict(entries)
    df.columns = [''.join(col).strip() for col in df.columns.values]
    selected_columns = ['key', 'compare']
    print(tabulate(df, headers='keys', showindex=False, tablefmt='psql'))


