from setuptools import Extension, setup
import numpy

ext_modules = [
    Extension(
        "csc.get_path_matrix",
        sources=[
            "src/csc/get_path_matrix.pyx"
        ],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]


if __name__ == "__main__":
    from Cython.Build import cythonize
    import numpy as np

    setup(
        ext_modules=cythonize(ext_modules, language_level="3"),
        include_dirs=[np.get_include()],
    )
