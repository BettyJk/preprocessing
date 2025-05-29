# setup.py
from setuptools import setup, find_packages

setup(
    name='data-preprocessing-pro',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'joblib'
    ],
    author='Bouthayna Jouak & Hajar EL Hadri',
    description='A Streamlit app for preprocessing datasets',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BettyJk/data-preprocessing-pro',  # adjust this to your repo
)
