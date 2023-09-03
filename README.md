# ScheMoE
ScheMoE

The development of this code refers to [tutel](https://github.com/microsoft/tutel).

## Prerequisite

torch==1.9.1

## How to install

```Shell
git clone https://github.com/Fragile-azalea/ScheMoE.git
cd ScheMoE
pip install -e .
```

## How to Use

```python3
# Single Machine:
 python3 -m torch.distributed.run --nproc_per_node=4 -m schemoe.examples.pre_test --batch_size=16
# Distribute:
# pls refers to schemoe/examples/run_mpi.sh
``` 

## Test Environment

- g++==7.5.0
- cuda==10.2 
- gpu==2080Ti

## Paper Link

https://dl.acm.org/doi/10.1145/3627703.3650083


