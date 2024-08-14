from setuptools import setup, find_packages

setup(
    name='FlashRank', 
    version='0.2.9', 
    packages=find_packages(),
    install_requires=[
        'tokenizers',
        'onnxruntime',
        'numpy',
        'requests',
        'tqdm',
        'dataclasses;python_version>="3.6" and python_version<"3.7"',  # dataclasses are included in python 3.7+
    ],  
    extras_require={
        'listwise': ['llama-cpp-python==0.2.76']
    },
    author='Prithivi Da',
    author_email='',
    description='Ultra lite & Super fast SoTA cross-encoder based re-ranking for your search & retrieval pipelines.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PrithivirajDamodaran/FlashRank',  
    license='Apache 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
