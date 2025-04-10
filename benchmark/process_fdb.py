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
from collections import defaultdict

misa_percent = 0
misa_count = 0
ck_percent = 0
ck_count = 0

# For forward convolution, we have:
# in_channels-in_h-in_w-fil-out_channels-out_h-out_w-batch_size-pad-stride-dilation-?-layout-data_type-direction
#
# For backward and backward weights, however, we have:
# out_channels-out_h-out_w-fil-in_channels-in_h-in_w-batch_size-pad-stride-dilation-?-layout-data_type-direction
#
# We define param_opts as for the forward convolution, and later exchange the in and out channels if we are processing
# a backward or backward weights convolution entry.
param_opts = [
    "in_channels", "in_h", "in_w", "fil",
    "out_channels", "out_h", "out_w", "batch_size",
    "pad", "stride", "dilation", None,
    "layout", None, "direction"
]
direction_dict = {"F": "--forw 1", "B": "--forw 2", "W": "--forw 4"}
layout_dict = {"NCHW": "", "NHWC": " --in_layout NHWC --fil_layout NHWC --out_layout NHWC"}

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
    param_vals = {}
    for i, opt in enumerate(param_opts):
        if opt:
            param_vals[opt] = key[i]

    for field in ["fil", "pad", "stride", "dilation"]:
        if field in param_vals:
            param_vals[field] = param_vals[field].split("x")

    layout = layout_dict.get(param_vals["layout"], "")
    conv_type = extract_conv_type(key)
    direction = direction_dict.get(param_vals["direction"], "-F 1")
    # If the entry is for a backward or backward weights convolution, in and out channels are exchanged
    if param_vals["direction"] == 'B' or param_vals["direction"] == 'W':
        out_channels = param_vals['in_channels']
        out_h = param_vals['out_h']
        out_w = param_vals['out_w']

        param_vals['in_channels'] = param_vals['out_channels']
        param_vals['in_h'] = param_vals['out_h']
        param_vals['in_w'] = param_vals['out_w']

        param_vals['out_channels'] = out_channels
        param_vals['out_h'] = out_h
        param_vals['out_w'] = out_w

    return (
        f"{args.miopen_cmd} {conv_type} "
        f"--batchsize {param_vals['batch_size']} --in_channels {param_vals['in_channels']} "
        f"--in_h {param_vals['in_h']} --in_w {param_vals['in_w']} "
        f"--out_channels {param_vals['out_channels']} "
        f"--fil_h {param_vals['fil'][0]} --fil_w {param_vals['fil'][1]} "
        f"--pad_h {param_vals['pad'][0]} --pad_w {param_vals['pad'][1]} "
        f"--conv_stride_h {param_vals['stride'][0]} --conv_stride_w {param_vals['stride'][1]} "
        f"--dilation_h {param_vals['dilation'][0]} --dilation_w {param_vals['dilation'][1]} "
        "--mode conv --group_count 1 "
        f"{direction}{layout} "
        "--time 1 "
    )

def add_algo_entry(algo, provider, time, solver, algos_dict):
    curr_time = algos_dict[algo][provider]['time']
    if not curr_time or time < curr_time:
        algos_dict[algo][provider]['time'] = time
        algos_dict[algo][provider]['solver'] = solver

def generate_json_entry(miopen_cmd_str, conv_args, arch, algo, conv_type, solver_name, time_ms):
    return ({
        'command': miopen_cmd_str,
        'args': conv_args,
        'device_arch': arch,
        'algo': algo,
        'results': [{
            'layer': conv_type,
            'solver_selected': solver_name,
            'read_bytes_mean': 0.0,
            'read_bytes_std': 0.0,
            'write_bytes_mean': 0.0,
            'write_bytes_std': 0.0,
            'mem_bw_gbs_mean': 0.0,
            'mem_bw_gbs_std': 0.0,
            'time_ms_mean': time_ms,
            'time_ms_std': 0.0
        }],
    })

def map_miopen_cmd_2_args(cmd):
    cmd_comp = cmd.split()
    conv_args = {
        comp[2:]: int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value
        for comp, value in zip(cmd_comp, cmd_comp[1:])
        if comp.startswith('--')
    }
    return conv_args

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
            conv_args = map_miopen_cmd_2_args(miopen_cmd_str)
            misa_data = generate_json_entry(miopen_cmd_str, conv_args, args.arch, algo, conv_type, misa_solver, misa_time)
            ck_data = generate_json_entry(miopen_cmd_str, conv_args, args.arch, algo, conv_type, ck_solver, ck_time)
            misa_dict.append(misa_data)
            ck_dict.append(ck_data)
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
