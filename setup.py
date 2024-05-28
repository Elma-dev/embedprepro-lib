from setuptools import setup, find_packages

setup(
    name='embedprepro',
    version='0.12',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='elmajjodi.abdeljalil@gmail.com',
    url='https://github.com/Elma-dev/embedprepro-lib.git/',
    packages=find_packages(),
    install_requires=[
        "click==8.1.7",
        "matplotlib==3.9.0",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "seaborn==0.13.2",
        "sentence_transformers==2.7.0",
        "setuptools==65.6.3",
        "tqdm==4.66.4",
        "umap_learn==0.5.6"
    ],
    entry_points={
        'console_scripts': [
            'embedprepro=preprocessing.cli:cli',
        ],
    },
)
