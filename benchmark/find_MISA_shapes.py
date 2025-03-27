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
import re
import json
from collections import deque

parser = argparse.ArgumentParser(
    prog='find_MISA_shapes',
    description='Find which MIOpenDriver invocations used MISA kernels from a set of logs')
parser.add_argument('-ld', '--logs-parent-dir', default=os.path.dirname(os.path.realpath(__file__)), help='Path to the parent directory of the logs folder (logs). E.g. /home/user/benchmark if the logs folder is /home/user/benchmark/logs')
parser.add_argument('-o', '--out-filename', default='fremont_conv_shapes_MISA_gt_CK', help='Name of the output JSON file (without .json extension)')

args = parser.parse_args()

script_dir=args.logs_parent_dir
logs_out_dir=f'{script_dir}/logs'
MISA_pattern = re.compile(r'registered as find 1\.0 best for (\w+) in .*?\nMIOpen\(HIP\): Info \[EvaluateInvokers\] Selected: ([a-zA-Z\d: _]*igemm[a-zA-Z\d: /_]*): ([0-9]+\.[0-9]+), workspace_sz = \d+')

MISA_shapes = {
    'conv' : {
      'bfp16': {
        'args': {},
        'include': []
        }
    }
}

for filename in os.listdir(logs_out_dir):
    log_path = os.path.join(logs_out_dir, filename)
    with open(log_path, 'r') as log:
        log_content = log.read()
        if MISA_pattern.search(log_content):
            with open(log_path, 'r') as temp_log:
                last_line = deque(temp_log, maxlen=1).pop().strip()
            args_line = last_line.split()
            args_dict = {}
            # process args skipping MIOpenDriver executable and layer name
            for i in range(2, len(args_line), 2):
                if i + 1 < len(args_line):
                    key = args_line[i].lstrip('--')
                    value = args_line[i + 1]
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                    args_dict[key] = value
            MISA_shapes['conv']['bfp16']['include'].append(args_dict)

with open(f'{script_dir}/{args.out_filename}.json', 'w') as f:
    json.dump(MISA_shapes, f, indent=2)
