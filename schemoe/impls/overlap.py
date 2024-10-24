# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from time import time

import schemoe_custom_kernel
import torch
from torch.distributed import get_rank

from ..impls import communicate as C


class Compress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, name):
        input = schemoe_custom_kernel.compress_operation(input, name, "naive")
        return input

    @staticmethod
    def backward(ctx, grad):
        grad = schemoe_custom_kernel.decompress_operation(grad)
        # print(grad)
        return grad, None


class Decompress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, name):
        ctx.name = name
        input = schemoe_custom_kernel.decompress_operation(input)
        return input

    @staticmethod
    def backward(ctx, grad):
        # grad = torch.ones_like(grad) * torch.distributed.get_rank() + torch.arange(0.0, 0.6, 0.1, device=grad.device) + torch.arange(0.00, 0.08, 0.01, device=grad.device).unsqueeze(1) + torch.arange(0.000, 0.002, 0.001, device=grad.device).unsqueeze(1).unsqueeze(1)
        # print(grad)
        return schemoe_custom_kernel.compress_operation(grad, ctx.name, "naive"), None


class Comm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return schemoe_custom_kernel.comm_operation(input)

    @staticmethod
    def backward(ctx, grad):
        return schemoe_custom_kernel.comm_operation(grad)


def a2a_ffn_overlap_forward(
    input, expert_fn, a2a_ffn_overlap_degree, use_2dh, group, is_compress
):
    split_dim = 1
    assert a2a_ffn_overlap_degree <= C.AllToAllStatus.max_num_split, (
        "Excepting a2a_ffn_overlap_degree (%d) <= AllToAllStatus.max_num_split (%d)."
        % (a2a_ffn_overlap_degree, C.AllToAllStatus.max_num_split)
    )
    assert (
        input.shape[split_dim] % a2a_ffn_overlap_degree == 0
    ), "Excepting input.shape[%d] (%d) be multiple of a2a_ffn_overlap_degree (%d)." % (
        split_dim,
        input.shape[split_dim],
        a2a_ffn_overlap_degree,
    )
    C.AllToAllStatus.init(group, a2a_ffn_overlap_degree, split_dim)
    name = "no"

    split_size = input.shape[split_dim] // a2a_ffn_overlap_degree
    input_split = list(input.split(split_size, dim=split_dim))
    schemoe_custom_kernel.clear_ptr_lst()
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = input_split[i].contiguous()
        input_split[i] = Compress.apply(input_split[i], name)
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = Comm.apply(input_split[i])

    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = Decompress.apply(input_split[i], name)
        input_split[i] = C.post_expert_permute(
            expert_fn(C.pre_expert_permute(input_split[i], group=group)), group=group
        )
        input_split[i] = Compress.apply(input_split[i], name)
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = Comm.apply(input_split[i])
    for i in range(a2a_ffn_overlap_degree):
        input_split[i] = Decompress.apply(input_split[i], name)
    output = torch.cat(input_split, dim=split_dim).contiguous()
    return output
