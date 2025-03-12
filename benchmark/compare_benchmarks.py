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
from tabulate import tabulate
from colorama import Fore, Style
from ansi2html import Ansi2HTMLConverter

parser = argparse.ArgumentParser(
    prog='compare_benchmarks',
    description='Compare MIOpen layers benchmark results')
parser.add_argument('files', nargs='*', type=str, help='JSON results')
parser.add_argument('-a', '--arch', default='gfx942', help='Architecture to target, default to gfx942')
parser.add_argument('-b', '--baseline-dir', help='Path to directory with baseline JSON files')
parser.add_argument('-c', '--compare-dir', help='Path to directory with JSON files to be compared against baseline')
parser.add_argument('-m', '--memory', action='store_true', default=False, help='Compare memory bandwidth')
parser.add_argument('-t', '--time', action='store_true', default=False, help='Compare execution time')
parser.add_argument('-oh', '--output-html', default='compare_benchmarks_output', help='Name of HTML tables output file (without extension)')

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
        for bresult, cresult in zip(bresults['results'], cresults['results']):
            if bresult['layer'] != cresult['layer']:
                print('Base and comparison results must contain the same layer+type combinations', file=sys.stderr)
                sys.exit(1)
            new_entry = {'layer': str(bresult['layer']), **{k: str(bresults['args'][k]) for k in keys}}
            for metric, mean_key, std_key in [('memory', 'mem_bw_gbs_mean', 'mem_bw_gbs_std'), ('time', 'time_ms_mean', 'time_ms_std')]:
                if getattr(args, metric):
                    if metric == 'memory':
                        diff = ((cresult[mean_key] - bresult[mean_key]) / bresult[mean_key] * 100) if bresult[mean_key] else 0
                    else:
                        diff = ((bresult[mean_key] - cresult[mean_key]) / bresult[mean_key] * 100) if bresult[mean_key] else 0
                    stddev = bresult[std_key]
                    color = pick_color(diff, stddev)
                    new_entry[f'{metric}_{diff_label}'] = f'{color}{diff}{Style.RESET_ALL}'
                    new_entry[f'{metric}_{std_label}'] = stddev
            entries.append(new_entry)

    columns = [('key', 'layer')]
    columns.extend([('key', k) for k in keys])
    columns.append(('compare', diff_label))
    columns.append(('compare', std_label))
    df = pd.DataFrame.from_dict(entries)
    df.columns = [''.join(col).strip() for col in df.columns.values]
    print(tabulate(df, headers='keys', showindex=False, tablefmt='psql'))
    # Also export to html for better visualization
    conv = Ansi2HTMLConverter(inline=True)
    df_html = df.map(lambda x: conv.convert(str(x), full=False))
    html_table = df_html.to_html(escape=False)
    with open(f'{args.output_html}_{args.arch}.html', 'w') as f:
        f.write(html_table)
