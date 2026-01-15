from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "linecounter_cpp",
        ["linecounter.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2", "/std:c++17"],
    )
]

setup(
    name="linecounter_cpp",
    version="0.0.1",
    ext_modules=ext_modules,
)
