# Benchmarking infrastructure

## Building MIOpen for benchmarking

The benchmarking script assumes an standarized statistics output across layers. This is not the default behaviour of `MIOpenDriver`, but it is enabled when the C++ macro `MIOPEN_PRINT_STD_STATS` is defined.

An homonymous CMake macro is also in place, so that when configuring the project with it defined the C++ macro will also be defined.

```shell
# Trigger 'MIOPEN_PRINT_STD_STATS' C++ macro definition at build time
cmake ... -DMIOPEN_PRINT_STD_STATS=ON ...
```

## Running the benchmarks

The [run_benchmarks.py](/benchmark/run_benchmarks.py) script allows to run a number of shapes for each of the desired `(layer,type)` pairs. These shapes are read from a JSON file, which path can be passed to the script as an argument. If no path is provided, by default the script will try to read a `bench_matrix.json` file from the same folder in which the script is placed (the `benchmark` folder).

Each layer to be benchmarked needs to have a set of types to benchmark for. For instance, the following JSON would allow to benchmark the convolution (`conv`) layer for types `fp16` and `bfp16`:

```json
{
    "conv" : {
        "fp16": {
            ...
        },
        "bfp16": {
            ...
        }
    }
}
```

For each `(layer,type)` pair, the shapes to benchmark can be described in two ways:

1. One way is by using a set of parameters and assigning a list of input values for each of them. For instance, as shown below.
    ```json
    {
        "conv" : {
            "fp16": {
                "args": {
                    "in_d": [in_d_0, in_d_1, ..., in_d_n],
                    "in_h": [in_h_0, in_h_1, ..., in_h_m],
                    "in_w": [in_w_0, in_w_1, ..., in_w_k]
                },
                "include": [
                    ...
                ]
            },
        }
    }
    ```
    The `run_benchmarks.py` can then be used to obtain the performance results of `convfp16` with the `(in_d, in_h, in_w)` input parameters taking values in the cartesian product of the lists specified in the `args` section of the JSON file.

2. A second way is by directly describing the specific shapes that should be benchmarked. For instance, as shown below.
    ```json
    {
        "conv" : {
            "fp16": {
                "args": {
                    ...
                },
                "include": [
                    {"in_d": in_d_0, "in_h": in_h_0, "in_w": in_w_0},
                    {"in_d": in_d_1, "in_h": in_h_1, "in_w": in_w_1},
                    {"in_d": in_d_2, "in_h": in_h_2, "in_w": in_w_2},
                ]
            },
        }
    }
    ```
    The `run_benchmarks.py` can then be used to obtain the performance results of `convfp16` with the `(in_d, in_h, in_w)` input parameters taking the values from each set (shape) specified in the `include` section of the JSON file.

The "mode" in which the `run_benchmarks.py` operates with respect to how shapes are read can be modified with the `--full-bench` (or `-f`) flag. By default, the second mode is enabled, and when running the script with the `--full-bench` flag the first mode is.

Both modes are mutually exclusive, only one of them can be enabled at a time.

### Results

The results for each layer are stored in a JSON file within a `benchmark_results` folder, located inside the specified output directory (`--out-dir`, or `-o` for short). If no output directory is provided, the results' folder is created within the script's parent folder by default.

The benchmarks for each `(layer, type, shape)` combination are run twice by default, and the mean and standard deviation metrics are reported within the results. If more iterations are desirable, the script accepts an `--iters` (or `-i`) flag for setting the number of iterations. Beware that the number of iterations should never be less than 2.

### Example

Assuming we'd like to benchmark the `conv` layer for the types `fp16` and `bfp16`, one possible JSON file for the benchmarking matrix could be:

```json
{
    "conv" : {
        "fp16": {
            "args": {
                "in_channels": [16,64,256],
                "in_d": [3,9,27],
                "in_h": [16,64,256],
                "in_w": [16,64,256],
                "out_channels": [32,128,512],
                "time": [1]
            },
            "include": [
                {"in_channels": 256,"in_d": 3,"in_h": 16,"in_w": 64,"out_channels": 512,"time": 1},
                {"in_channels": 256,"in_d": 9,"in_h": 16,"in_w": 64,"out_channels": 512,"time": 1},
                {"in_channels": 256,"in_d": 27,"in_h": 16,"in_w": 64,"out_channels": 512,"time": 1}
            ]
        },
        "bfp16": {
            "args": {
                "in_channels": [16,64,256],
                "in_d": [3,9,27],
                "in_h": [16,64,256],
                "in_w": [16,64,256],
                "out_channels": [32,128,512],
                "time": [1]
            },
            "include": [
                {"in_channels": 16,"in_d": 27,"in_h": 256,"in_w": 256,"out_channels": 512, "time": 1},
                {"in_channels": 64,"in_d": 27,"in_h": 256,"in_w": 256,"out_channels": 512, "time": 1},
                {"in_channels": 256,"in_d": 27,"in_h": 256,"in_w": 256,"out_channels": 512, "time": 1}
            ]
        }
    }
}
```

And we could either get the results for the whole cartesian product of the parameters' values from the `args` sections:

```shell
# From project's root folder
python3 benchmark/run_benchmarks.py --full-matrix
```

or just get them for the interesting shapes:

```shell
# From project's root folder
python3 benchmark/run_benchmarks.py
```

#### Using custom-built MIOpenDriver

In case one may want to benchmark some custom-built `MIOpenDriver` executable instead of the one shipped with the MIOpen library, the script can also be invoked as follows:

```shell
# From project's root folder
python3 benchmark/run_benchmarks.py --miopen-cmd "<path-to-build-folder>/bin/MIOpenDriver" --full-matrix
python3 benchmark/run_benchmarks.py --miopen-cmd "<path-to-build-folder>/bin/MIOpenDriver"
```

### Other available options

For a complete list of the available flags for the `run_benchmarks.py` script, please invoke the help message as shown below.

```shell
python3 run_benchmarks.py -h
```

## Comparing benchmarks results

Once gathered the performance results, they can be compared to previous results with the [compare_benchmarks.py](/benchmark/compare_benchmarks.py) script.

This script allows to compare all the results from certain _baseline_ and _compare_ folders, that can be specified with the `--baseline` and `--compare` flags (or `-b` and `-c` for the short form).

Alternatively, one can pass the list of files to be compared, being the first half of the list the baseline results and the second half of the list the results to be compared.

In both cases, the script assumes (and checks) that there is the same amount of baseline and compare files.

The two sets of files are then ordered by filename and processed in pairs. That means that the first baseline file (in alphabetical order) and the first compare file will be processed together. The script thus assumes (and checks) that each pair of such files contain the same `(layer, type)` pairs and the same shapes.

### Comparison report

The script outputs several reports summarizing the performance differences between the two sets of results.

Two performance metrics can be included in such reports: execution __time__ and __memory__ bandwith. At least one of them must be enabled when invoking the script, with the `--time` or `--memory` flags, respectivelly. Both can also be enabled simultanously.

The script will then print to standard output a table with the summary of the results, including the information of the layers' names, types, arguments (shapes), the mean execution time or/and memory bandwith difference with respect to the baseline results (in % of the baseline metric/s value/s) and the standard deviation (in % of the mean).

The performance differences will also be printed in different colours to highlight whether a certain result corresponds to an improvement, a regression or are subject to interpretation. We assume three possible outcomes from the performance comparison:

1. the new result is more than $1%$ worse than the baseline result,
2. the new result is less than $1%$ worse than the baseline result (but still there is a regression greater than $0%$),
3. or the new results is equal or better than the baseline result.

- __yellow__ is used when in case 2 or when in case 1 but the standard deviation of the new results is greater or equal than the regression obtained,
- __red__ is used when in case 1,
- __green__ in used otherwise (case 3).

Additionally, an HTML file is output with the same table/information as above for an easier visualization of the data.

### Other available options

For a complete list of the available flags for the `compare_benchmarks.py` script, please invoke the help message as shown below.

```shell
python3 compare_benchmarks.py -h
```

### Example

Assuming we have our baseline results in a folder `base_dir` and our new results in a folder `new_dir`, we showcase several ways in which the benchmark comparison script can be invoked.

```shell
# Invoke the script passing the baseline and comparison folders as arguments. Report execution time differences
python3 compare_benchmarks.py --time -b base_dir/ -c new_dir/

# Same as before, but reporting memory bandwidth differences instead
python3 compare_benchmarks.py --memory -b base_dir/ -c new_dir/

# Same as previous, but reporting both time and memory bandwidth differences
python3 compare_benchmarks.py --time --memory -b base_dir/ -c new_dir/

# Invoke the script passing the baseline and comparison results JSON files directly as arguments.
# Report execution time differences
python3 compare_benchmarks.py --time base_dir/benchmark_bnorm.json new_dir/benchmark_bnorm.json

# Invoke the script passing the baseline and comparison results JSON files directly as arguments.
# Report memory bandwidth differences
# * Note that the first half of the files are the baseline ones, and the second half are the new
#   ones.
# * The order is also important, we want new_dir/benchmark_bnorm.json to be compared with
#   base_dir/benchmark_bnorm.json and the script will process the benchmarks in pairs, picking up
#   one benchmark from each ordered list (base and new) at a time. So the results files, although
#   can have different naming conventions, should preserve the (ascending alphabetical) order
#   within its folder.
python3 compare_benchmarks.py --memory base_dir/benchmark_bnorm.json base_dir/benchmark_conv.json new_dir/benchmark_bnorm.json new_dir/benchmark_conv.json
```
