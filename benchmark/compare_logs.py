#!/usr/bin/env python3

# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import re
import argparse
from collections import deque

parser = argparse.ArgumentParser(
    prog='compare_logs',
    description='Compare logs from MISA and CK kernels',
)

parser.add_argument('-b','--bandwidth-gbs', type=float, default=800.0, help='Max theoretical bandwidth of the GPU in GB/s, default to 800GB/s')
parser.add_argument('-m', '--misa-dir', help='Path to directory with MISA (baseline) log files')
parser.add_argument('-ck', '--ck-dir', help='Path to directory with CK log files to be compared against misa')
parser.add_argument('-mo', '--misa-output-json', default='misa_conv_bfp16_gfx90a', help='Name of JSON output file (without extension) for selected MISA kernels results')
parser.add_argument('-cko', '--ck-output-json', default='ck_conv_bfp16_gfx90a', help='Name of JSON output file (without extension) for selected CK kernels')
parser.add_argument('-o', '--out-dir', default=f'{os.path.dirname(os.path.realpath(__file__))}/logs_comparing_results' ,help='Path to directory for output JSON files')

args = parser.parse_args()

def extract_device_arch(log_content):
    match = re.search(r'Raw device name: (\S+)', log_content)
    return match.group(1).split(':')[0] if match else 'unknown'

def extract_command(log_path):
    with open(log_path, 'r') as temp_log:
        return deque(temp_log, maxlen=1).pop().strip()

def extract_shape(driver_command_str):
    cmd_args = {}
    matches = re.findall(r'--(\w+)\s+(\S+)', driver_command_str)
    for key, value in matches:
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        cmd_args[key] = value
    return cmd_args

def process_log_file(log_path, algo_name=None):
    with open(log_path, 'r') as log:
        log_content = log.read()

    device_arch = extract_device_arch(log_content)
    driver_command_str = extract_command(log_path)
    shape = extract_shape(driver_command_str)

    # For logs from MISA-enabled-MIOpen executions, we look for the MISA kernel selected (if any)
    if not algo_name:
        algo_pattern = re.compile(r'registered as find 1\.0 best for (\w+) in .*?\nMIOpen\(HIP\): Info \[EvaluateInvokers\] Selected: ([a-zA-Z\d: _]*igemm[a-zA-Z\d: /_]*): ([0-9]+\.[0-9]+), workspace_sz = \d+')
        match = algo_pattern.search(log_content)
        if not match:
            return None
        algo_name = match.group(1)
        kernel_name = match.group(2)
        time_ms_mean = float(match.group(3))
    # For logs from MISA-disabled-MIOpen executions, we look for the CK kernel selected instead for a particular algo
    else:
        algo_pattern = re.compile(rf'registered as find 1\.0 best for {algo_name} in .*?\nMIOpen\(HIP\): Info \[EvaluateInvokers\] Selected: ([a-zA-Z\d: /_]*): ([0-9]+\.[0-9]+), workspace_sz = \d+')
        match = algo_pattern.search(log_content)
        if not match:
            return None
        kernel_name = match.group(1)
        time_ms_mean = float(match.group(2))

    shape['algo'] = algo_name

    return {
        'command': driver_command_str,
        'args': shape,
        'device_arch': device_arch,
        'bandwidth_gbs': args.bandwidth_gbs,
        'results': [{
            'layer': 'bwdd-convbfp16',
            'kernel_selected': kernel_name,
            'read_bytes_mean': 0.0,
            'read_bytes_std': 0.0,
            'write_bytes_mean': 0.0,
            'write_bytes_std': 0.0,
            'mem_bw_gbs_mean': 0.0,
            'mem_bw_gbs_std': 0.0,
            'time_ms_mean': time_ms_mean,
            'time_ms_std': 0.0
        }],
    }

def process_logs(misa_log_dir, ck_log_dir, misa_output_json, ck_output_json):
    misa_output_json_data = []
    ck_output_json_data = []
    algo_times = {}

    # Process the MISA-enabled-MIOpen results
    for filename in os.listdir(misa_log_dir):
        filepath = os.path.join(misa_log_dir, filename)
        if not os.path.isfile(filepath):
            continue
        data = process_log_file(filepath)
        if data:
            misa_output_json_data.append(data)
            algo_times[data['command']] = data['args']['algo']

    # Process the MISA-disabled-MIOpen results using 'algo's from previous processing
    for filename in os.listdir(ck_log_dir):
        filepath = os.path.join(ck_log_dir, filename)
        if not os.path.isfile(filepath):
            continue
        driver_command_str = extract_command(filepath)
        algo_name = algo_times[driver_command_str]
        data = process_log_file(filepath, algo_name)
        if data:
            ck_output_json_data.append(data)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.out_dir}/{misa_output_json}', 'w') as f:
        json.dump(misa_output_json_data, f, indent=2)
    with open(f'{args.out_dir}/{ck_output_json}', 'w') as f:
        json.dump(ck_output_json_data, f, indent=2)

process_logs(args.misa_dir, args.ck_dir, f'{args.misa_output_json}.json', f'{args.ck_output_json}.json')
