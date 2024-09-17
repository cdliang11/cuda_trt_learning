import torch
from torch import Tensor

import triton  # 只能用于 主机函数
import triton.language as tl  # 只能用于 核函数


@triton.jit
def _vector_add_kernel(
        a_ptr, b_ptr, c_ptr, M,        # c = a + b, 其中, a, b 和 c 的 shape 都是 [M, ]
        stride_a, stride_b, stride_c,  # a, b, c 向量的 stride
        BLOCK_SIZE: tl.constexpr       # meta 参数
):
    """ 每一个 program 处理 [BLOCK_SIZE, ] 大小的数据 """

    # pid: 当前 program 的索引值
    pid = tl.program_id(axis=0)

    # offsets: 当前 program 需要处理元素的 坐标索引
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # ptrs: 当前 program 需要处理元素的内存地址
    a_ptrs = a_ptr + offsets * stride_a
    b_ptrs = b_ptr + offsets * stride_b

    # 加载数据并计算
    a = tl.load(a_ptrs, mask=offsets < M, other=0.0)
    b = tl.load(b_ptrs, mask=offsets < M, other=0.0)
    c = a + b

    # 保存数据
    c_ptrs = c_ptr + offsets * stride_c
    tl.store(c_ptrs, c, mask=offsets < M)


def vector_add(
        input1: Tensor, input2: Tensor, block_size: int = 1024,
        print_ir: bool = False, ir_type: str = "llir", print_meta: bool = False
) -> Tensor:

    assert input1.is_cuda and input2.is_cuda
    assert input1.ndim == 1 and input2.ndim == 1 and input1.size(0) == input2.size(0)

    vector_size = input1.size(0)
    output = torch.zeros_like(input1)

    block_size = triton.next_power_of_2(block_size)
    num_programs = triton.cdiv(vector_size, block_size)
    programs_shape = (num_programs, )

    compiled_kernel: triton.compiler.CompiledKernel = _vector_add_kernel[programs_shape](
        input1, input2, output, vector_size, input1.stride(0), input2.stride(0), output.stride(0), block_size
    )

    if print_ir:
        print("Triton GPU IR codes of add kernel are shown below:", )
        print(compiled_kernel.asm[ir_type])
        print()

    if print_meta:
        print("The meta parameters of add kernel are shown below:",)

        while True:  # 等待程序运行完成
            if isinstance(compiled_kernel.metadata, dict):
                print(json.dumps(compiled_kernel.metadata, ensure_ascii=False, indent=4, skipkeys=True))
                break

    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # size 是 x 轴
        x_vals=[2 ** i for i in range(12, 28, 1)],  # 从 2^12 到 2^28
        x_log=True,  # x 轴使用 log10 显示
        line_arg="provider",
        line_vals=["triton", "torch"],  # 一条曲线
        line_names=["Triton", "Torch"],  # 曲线名称
        styles=[('blue', '-'), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector_add_performance",
        args={}
    ))


def benchmark(size, provider):
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y,)
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add(x, y),)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-6 / ms

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output_torch = x + y
    output_triton = vector_add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

    benchmark.run(print_data=True, save_path="vector_add_performance/")
