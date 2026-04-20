from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).parent
source_dir = ROOT / "lib" / "cpp_src"

ext_modules = [
    Pybind11Extension(
        "lib.cpp_ext.cpp_chi_square_distance",
        [str(source_dir / "chi_square_distance.cpp")],
        extra_compile_args=["-O3"],
        cxx_std=11,
    ),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
