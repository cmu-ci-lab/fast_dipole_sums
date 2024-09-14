from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='interp_geometry_cuda',
    ext_modules=[
        CUDAExtension('interp_geometry_cuda', [
            'interp_cuda.cpp',
            'kernel_utils.cu',
            'interp_cuda_forward_kernel.cu',
            'interp_cuda_backward_kernel.cu',
            'interp_cuda_gradient_kernel.cu',
            'interp_cuda_init_kernel.cu'
        ],
        extra_compile_args={'cxx': ['-O3', '-fopenmp'], 'nvcc': ['-O3']},
        extra_link_args=['-lgomp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
