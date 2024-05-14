from setuptools import setup, find_packages

install_requires = [
    "numpy >= 1.24.3",
    "pandas >= 1.5.3",
    "matplotlib >= 3.7.1",
    "seaborn >= 0.12.2",
    'pytest>=6.2.4',
    'scipy>=1.7.1',
    'torch>=1.12.1',
    'pytorch-lightning==1.6.0',
    'torchdiffeq==0.2.3',
    'torchsde==0.2.5',
    'matplotlib>=3.4.3',
    'seaborn==0.11.1',
    'pytorchts==0.6.0',
    'gluonts==0.9.*',
    'wget==3.2',
]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='DSPD_CW',
      version='1.0.0',
      description='DSPD time series forecasting and imputation',
      long_description='DSPD time series forecasting and imputation',
      long_description_content_type='text/markdown',
      url='',
      author='Artyom Kraevskiy',
      author_email='akraevskiy@hse.ru',
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.9',
      zip_safe=False,
)
