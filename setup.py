import os
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

def get_extensions():

    # prevent ninja from using too many resources
    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

    os.environ.setdefault('MAX_JOBS', str(cpu_use))

    extra_compile_args = {}
    if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        
        # objc compiler support
        from distutils.unixccompiler import UnixCCompiler
        if '.mm' not in UnixCCompiler.src_extensions:
            UnixCCompiler.src_extensions.append('.mm')
            UnixCCompiler.language_map['.mm'] = 'objc'

        extra_compile_args = {}
        extra_compile_args['cxx'] = [
            '-Wall', 
            '-std=c++17',
            '-framework', 
            'Metal', 
            '-framework', 
            'Foundation',
            '-ObjC++'
            ]
    else:
        extra_compile_args['cxx'] = [
            '-std=c++17'
            ]

    ext_ops = CppExtension(
        name='my_extension_cpp',
        sources=['my_extension/cpp_extension.mm'],
        include_dirs=[],
        extra_objects=[],
        extra_compile_args=extra_compile_args,
        library_dirs=[],
        libraries=[],
        extra_link_args=[]
        )
    return [ext_ops]


setup(
    name='my_extension',
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.11',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    requires=[
        'torch',
        'setuptools'
        ]
)