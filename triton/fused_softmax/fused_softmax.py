
import torch
from torch import Tensor

import triton  # 只能用于 主机函数
import triton.language as tl  # 只能用于 核函数

def naive_softmax(input: Tensor):
    assert input.ndim == 2
    output = input - input.max(dim=1, keepdim=True)[0]
    output = torch.exp(output)
    output = output / output.sum(dim=1, keepdim=True)
    return output


# @triton.autotune(configs=[triton.Config({}, num_warps=2 ** i) for i in range(0, 6)], key=["BLOCK_SIZE_N"])
@triton.jit
def _softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # softmax 的行是独立的，所以我们在这些行上并行化
    row_idx = tl.program_id(0)
    # 步长表示我们需要增加指针的数量以前进1行
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # 块大小是大于 n_cols 的下一个2的幂，这样我们可以将每
    # 行适配在单个块中
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # 使用掩码将行加载到SRAM中，因为 BLOCK_SIZE 可能大于 n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # 减去最大值以保证数值稳定性
    row_minus_max = row - tl.max(row, axis=0)
    # 注意，在 Triton 中指数运算是快速但近似的（即，想象在 CUDA 中的 __expf）
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # 将输出写回到 DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def fused_softmax(x):
    n_rows, n_cols = x.shape
    # 块大小是大于 `x` 中列数的最小2的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 另一个我们可以使用的技巧是要求编译器通过
    # 增加每行分布的 warps 数量（`num_warps`）来使用更多线程。
    # 在下一个教程中，你将看到如何以更自然的方式自动调整这个值，
    # 这样你就不必自己提出手工启发式方法。
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # 分配输出
    y = torch.empty_like(x)
    # 排队内核。1D启动网格很简单：输入矩阵的每一行分配一个 kernel 实例
    _softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作图表x轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # `x_name`的不同可能值
        line_arg='provider',  # 其值对应图表中不同线条的参数名
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],  # `line_arg`的可能值
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],  # 线条的标签名
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # 线条样式
        ylabel="GB/s",  # y轴的标签名
        plot_name="softmax-performance",  # 图表的名称。也用作保存图表的文件名。
        args={'M': 4096},  # 不在`x_names`和`y_name`中的函数参数值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    # quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: fused_softmax(x))
    if provider == 'torch-jit':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = fused_softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    benchmark.run(save_path="softmax_performance/", print_data=True)
