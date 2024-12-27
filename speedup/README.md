# Speedup Evaluation
This directory contains source codes for evaluating the speedup. You can reproduce inference latency results in the paper. Some of the codes are referenced from FlexGen (ICML'23) GitHub repository.
- Getting Started (10 minutes)
- Run Experiments (7 hours)

## Getting Started (10 minutes)
```sh
sh install.sh
export CUDA_HOME=/path/to/cuda
```
For a "Hello world"-sized example, please run the following command (10 minutes):
```
# Original OPT. It might not work for this repository
python -m flexgen.flex_opt --model huggingface/opt-6.7b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 2 --num-gpu-batches 1 --prompt-len 384 --gen-len 128 --warmup-input-path flexgen/pg19_firstbook.txt --test-input-path flexgen/pg19_firstbook.txt --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 102

# Llama-3.1 Support
python -m flexgen.flex_llama --model flexgen/flexgen/InfiniGen-Meta-Llama-3.1-8B-Instruct --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 4096 --gen-len 128 --warmup-input-path flexgen/pg19_firstbook.txt --test-input-path flexgen/pg19_firstbook.txt --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 102

# Llama-3.1 + KVQuant use 0.01% sparsity 
python -m flexgen.flex_llama_offload --model flexgen/flexgen/InfiniGen-Meta-Llama-3.1-8B-Instruct --percent 100 0 100 100 100 0 --overlap false --gpu-batch-size 5 --num-gpu-batches 1 --prompt-len 1024 --gen-len 128 --warmup-input-path flexgen/pg19_firstbook.txt --test-input-path flexgen/pg19_firstbook.txt --quantizer flexgen/flexgen/norm_gsm8k_cot_quantizers/llama-3.1-8b-quantizer-2-0.01.pickle --sparsity-threshold 0.99
```
## Run Experiments (7 hours)
We provide scripts to reproduce the experiment results from Figure 14 to Figure 17. To reproduce all the results at once, please run the following commands (7 hours).
```
cd scripts
sh run_all.sh
```
If you want to reproduce the results for a specific figure, please `sh run.sh` in each corresponding directory. For example,
```
cd scripts/figure14
sh run.sh
```
Following is the amount of time to run each experiments on our system (NVIDIA RTX A6000 GPU with 48GB of memory, Intel Xeon Gold 6136 processor with 96GB of DDR4-2666 memory, PCIe 3.0 x16 interconnection).
- Figure 14: 80 minutes
- Figure 15: 220 minutes
- Figure 16a: 30 minutes
- Figure 16b: 60 minutes
- Figure 17a: 20 minutes
- Figure 17b: 10 minutes

