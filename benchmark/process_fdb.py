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

import argparse
import os
import sys
import json
import logging
import rich.logging
import rich.progress
from typing import Dict

misa_percent = 0
misa_count = 0
ck_percent = 0
ck_count = 0

def extract_conv_type(key):
    data_type = key[13]
    if data_type == "FP32":
        conv_type = "conv"
    if data_type == "FP16":
        conv_type = "convfp16"
    if data_type == "BF16":
        conv_type = "convbfp16"
    return conv_type
    
def make_2d_cmd(key):
    in_channels = key[0]
    in_h = key[1]
    in_w = key[2]
    
    weights = key[3].split("x")
    weight_height = weights[0]
    weight_width = weights[1]
    
    out_channels = key[4]
    # 5 and 6 is in_h and in_w again
    batch_size = key[7]
    
    padding = key[8].split("x")
    pad_height = padding[0]
    pad_width = padding[1]
    
    strides = key[9].split("x")
    stride_height = strides[0]
    stride_width = strides[1]
    
    dilations = key[10].split("x")
    dilation_height = dilations[0]
    dilation_width = dilations[1]
    
    layout = key[12]
    conv_type = extract_conv_type(key)
    direction = key[14]

    if direction == "F":
        arg_dir = "-F 1"
    if direction == "B":
        arg_dir = "-F 2"
    if direction == "W":
        arg_dir = "-F 4"

    if layout == "NCHW":
        arg_layout = ""
    if layout == "NHWC":
        arg_layout = " --in_layout NHWC --fil_layout NHWC --out_layout NHWC"

    return f"{args.miopen_cmd} {conv_type} -n {batch_size} -c {in_channels} -H {in_h} -W {in_w} -k {out_channels} -y {weight_height} -x {weight_width} -p {pad_height} -q {pad_width} -u {stride_height} -v {stride_width} -l {dilation_height} -j {dilation_width} {arg_dir}{arg_layout} $ARGS"

def add_algo_entry(algo, provider, time, solver, algos_dict):
    curr_time = algos_dict[algo][provider]['time']
    if not curr_time or time < curr_time:
        algos_dict[algo][provider]['time'] = time
        algos_dict[algo][provider]['solver'] = solver

def process_2d(shape, solvers, misa_dict, ck_dict):
    global misa_percent
    global misa_count
    global ck_percent
    global ck_count

    algos_data = {}
    for solver in solvers:
        solver_data = solver.split(":")
        solver_name = solver_data[0]
        fields = solver_data[1].split(",")
        time = float(fields[0])
        algo = fields[2]
        if not algo in algos_data:
            algos_data[algo] = {'misa': {'time': 0.0, 'solver': ""}, 'ck': {'time': 0.0, 'solver': ""}}
        if solver_name.startswith("ConvAsm"):
            add_algo_entry(algo, 'misa', time, solver_name, algos_data)
        if solver_name.startswith("ConvHip"):
            add_algo_entry(algo, 'ck', time, solver_name, algos_data)
    for algo in algos_data.keys():
        ck_time = algos_data[algo]['ck']['time']
        ck_solver = algos_data[algo]['ck']['solver']
        misa_time = algos_data[algo]['misa']['time']
        misa_solver = algos_data[algo]['misa']['solver']
        if ck_time and misa_time:
            if misa_time < ck_time:
                misa_percent += (ck_time - misa_time) / ck_time
                misa_count += 1
            else:
                ck_percent += (misa_time - ck_time) / misa_time
                ck_count += 1
            miopen_cmd_str = make_2d_cmd(shape)
            conv_type = extract_conv_type(shape)
            misa_data = {
                'command': miopen_cmd_str,
                'args': shape,
                'device_arch': args.arch,
                'algo': algo,
                'results': [{
                    'layer': conv_type,
                    'solver_selected': misa_solver,
                    'read_bytes_mean': 0.0,
                    'read_bytes_std': 0.0,
                    'write_bytes_mean': 0.0,
                    'write_bytes_std': 0.0,
                    'mem_bw_gbs_mean': 0.0,
                    'mem_bw_gbs_std': 0.0,
                    'time_ms_mean': misa_time,
                    'time_ms_std': 0.0
                }],
            }
            ck_data = {
                'command': miopen_cmd_str,
                'args': shape,
                'device_arch': args.arch,
                'algo': algo,
                'results': [{
                    'layer': conv_type,
                    'solver_selected': ck_solver,
                    'read_bytes_mean': 0.0,
                    'read_bytes_std': 0.0,
                    'write_bytes_mean': 0.0,
                    'write_bytes_std': 0.0,
                    'mem_bw_gbs_mean': 0.0,
                    'mem_bw_gbs_std': 0.0,
                    'time_ms_mean': ck_time,
                    'time_ms_std': 0.0
                }],
            }
            misa_dict.append(misa_data)
            ck_dict.append(ck_data)
            print(f"{miopen_cmd_str}\t{misa_time}\t{ck_time}\t{algo}")
    else:
        if(args.verbose):
            log.info(f'DB entry {shape}={solvers} doesn\'t contain both MISA and CK solvers for the same algo')

def process_line(line, misa_dict, ck_dict):
    db_entry = line.split("=")
    shape = db_entry[0].split("-")
    solvers = db_entry[1].split(";")

    if len(shape) == 15:
        process_2d(shape, solvers, misa_dict, ck_dict)
    else:
        if(args.verbose):
            log.info(f'DB entry {db_entry} is not for 2D convolution')

def process_fdb(file):
    misa_output_json_data = []
    ck_output_json_data = []
    for line in file:
        line = line.strip()
        if line:
            process_line(line, misa_output_json_data, ck_output_json_data)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.out_dir}/{args.misa_output_json}_{args.arch}.json', 'w') as f:
        json.dump(misa_output_json_data, f, indent=2)
    with open(f'{args.out_dir}/{args.ck_output_json}_{args.arch}.json', 'w') as f:
        json.dump(ck_output_json_data, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='process_fdb',
        description='Get MISA vs CK performance from UserDB entries',
    )

    parser.add_argument('-a', '--arch', default='gfx942', help='Architecture to target, default to gfx942')
    parser.add_argument('-f', '--fdb-path', help='Path to the MIOpenDriver fdb file')
    parser.add_argument('-m', '--miopen-cmd', default='MIOpenDriver', help='MIOpen command, default to system installation')
    parser.add_argument('-mo', '--misa-output-json', default='misa_conv_bfp16', help='Name of JSON output file (without extension) for selected MISA kernels results')
    parser.add_argument('-cko', '--ck-output-json', default='ck_conv_bfp16', help='Name of JSON output file (without extension) for selected CK kernels')
    parser.add_argument('-o', '--out-dir', default=f'{os.path.dirname(os.path.realpath(__file__))}/misa_vs_ck_output' ,help='Path to directory for output JSON files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(format='%(message)s', handlers=[rich.logging.RichHandler(rich_tracebacks=True, markup=True)], level=log_level)
    log = logging.getLogger('rich')

    try:
        with open(args.fdb_path, "r") as fdb:
            log.info(f'Processing {args.fdb_path} file for {args.arch}')
            process_fdb(fdb)
    except Exception as e:
        log.error(f'Could not process fdb file {args.fdb_path}: {e}')
        sys.exit(1)
