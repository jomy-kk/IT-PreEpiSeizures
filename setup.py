from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

requirements = [
    'alabaster==0.7.12',
    'altair==4.2.0',
    'appdirs==1.4.4',
    'appnope==0.1.3',
    'argon2-cffi==21.3.0',
    'argon2-cffi-bindings==21.2.0',
    'astroid==2.11.5',
    'asttokens==2.0.5',
    'attrs==21.4.0',
    'Babel==2.10.3',
    'backcall==0.2.0',
    'backoff==1.11.1',
    'beautifulsoup4==4.11.1',
    'bidict==0.22.0',
    'biosppy==0.8.0',
    'bleach==5.0.1',
    'bokeh==2.4.3',
    'boltons==21.0.0',
    'certifi==2021.10.8',
    'cffi==1.15.0',
    'cftime==1.6.3',
    'chardet==4.0.0',
    'charset-normalizer==2.0.12',
    'click==8.1.3',
    'click-spinner==0.1.10',
    'colorlog==6.7.0',
    'colorspacious==1.1.2',
    'commonmark==0.9.1',
    'cramjam==2.5.0',
    'cycler==0.11.0',
    'Cython==0.29.32',
    'dacite==1.6.0',
    'darkdetect==0.8.0',
    'datacommons==1.4.3',
    'datacommons-pandas==0.0.3',
    'datapane==0.15.3',
    'DateTime==5.1',
    'DateTimeRange==1.2.0',
    'debugpy==1.6.0',
    'decorator==5.1.1',
    'defusedxml==0.7.1',
    'diagrams==0.21.1',
    'dill==0.3.4',
    'docutils==0.18.1',
    'dominate==2.7.0',
    'dulwich==0.20.46',
    'edflib==0.84.1',
    'EDFlib-Python==1.0.6',
    'entrypoints==0.4',
    'executing==0.8.3',
    'face==20.1.1',
    'fastjsonschema==2.15.3',
    'fastparquet==0.8.3',
    'flake8==5.0.4',
    'fonttools==4.33.2',
    'fpdf2==2.5.5',
    'fsspec==2022.8.2',
    'furl==2.1.3',
    'furo==2022.12.7',
    'future==0.18.2',
    'gitdb==4.0.9',
    'GitPython==3.1.27',
    'glom==22.1.0',
    'graphviz==0.19.2',
    'h5py==3.7.0',
    'idna==3.3',
    'imagesize==1.4.1',
    'importlib-metadata==4.12.0',
    'importlib-resources==5.9.0',
    'infinity==1.5',
    'iniconfig==1.1.1',
    'ipykernel==6.15.0',
    'ipython==8.4.0',
    'ipython-genutils==0.2.0',
    'ipywidgets==7.7.1',
    'isort==5.10.1',
    'jedi==0.18.1',
    'Jinja2==3.1.1',
    'joblib==1.1.0',
    'jsonschema==4.6.0',
    'jupyter==1.0.0',
    'jupyter-client==7.3.4',
    'jupyter-console==6.4.4',
    'jupyter-core==4.10.0',
    'jupyterlab-pygments==0.2.2',
    'jupyterlab-widgets==1.1.1',
    'keyring==23.7.0',
    'kiwisolver==1.4.2',
    'lazy-object-proxy==1.7.1',
    'lazy_loader==0.3',
    'littleutils==0.2.2',
    'lxml==4.9.1',
    'markdown-it-py==2.1.0',
    'MarkupSafe==2.1.1',
    'matplotlib==3.5.1',
    'matplotlib-inline==0.1.3',
    'mbstrdecoder==1.1.0',
    'mccabe==0.7.0',
    'mdit-py-plugins==0.3.3',
    'mdurl==0.1.2',
    'micawber==0.5.4',
    'mistune==0.8.4',
    'mne',
    'mne-connectivity',
    'mne-qt-browser==0.6.1',
    'monotonic==1.6',
    'multimethod==1.9.1',
    'multipledispatch==0.6.0',
    'munch==2.5.0',
    'myst-parser==0.18.1',
    'nbclient==0.6.4',
    'nbconvert==6.5.0',
    'nbformat==5.4.0',
    'neo==0.10.2',
    'nest-asyncio==1.5.5',
    'netCDF4==1.6.5',
    'notebook==6.4.12',
    'numexpr==2.8.3',
    'numpy==1.22.3',
    'opencv-python==4.5.5.64',
    'orderedmultidict==1.0.1',
    'outdated==0.2.2',
    'packaging==21.3',
    'pandas==1.4.2',
    'pandas-flavor==0.6.0',
    'pandocfilters==1.5.0',
    'parso==0.8.3',
    'patsy==0.5.2',
    'pexpect==4.8.0',
    'pickleshare==0.7.5',
    'Pillow==9.1.0',
    'pingouin==0.5.3',
    'pkginfo==1.8.3',
    'platformdirs==2.5.2',
    'plotly==5.10.0',
    'plotly-express==0.4.1',
    'pluggy==1.0.0',
    'ply==3.11',
    'pooch==1.6.0',
    'posthog==2.1.2',
    'prometheus-client==0.14.1',
    'prompt-toolkit==3.0.30',
    'psutil==5.9.1',
    'ptyprocess==0.7.0',
    'pure-eval==0.2.2',
    'py==1.11.0',
    'py-cpuinfo==8.0.0',
    'pyarrow==6.0.1',
    'pycodestyle==2.9.1',
    'pycparser==2.21',
    'pydantic==1.10.2',
    'pyEDFlib==0.1.30',
    'pyflakes==2.5.0',
    'Pygments==2.12.0',
    'pylint==2.13.9',
    'pymatreader==0.0.32',
    'Pympler==1.0.1',
    'PyOpenGL==3.1.7',
    'pyparsing==3.0.8',
    'PyQt5',
    'PyQt5-Qt5',
    'PyQt5-sip',
    'pyqtgraph',
    'pyrsistent==0.18.1',
    'pytest==7.1.2',
    'python-dateutil==2.8.2',
    'pytz==2022.1',
    'PyYAML==6.0',
    'pyzmq==23.2.0',
    'QDarkStyle==3.2.3',
    'qtconsole==5.3.1',
    'QtPy==2.1.0',
    'quantities==0.13.0',
    'readme-renderer==35.0',
    'requests==2.27.1',
    'requests-toolbelt==0.9.1',
    'rfc3986==2.0.0',
    'rich==12.5.1',
    'scikit-learn==1.1.0',
    'scipy==1.8.0',
    'scooby==0.9.2',
    'seaborn==0.11.2',
    'Send2Trash==1.8.0',
    'shortuuid==1.0.9',
    'six==1.16.0',
    'smmap==5.0.0',
    'snowballstemmer==2.2.0',
    'sortedcontainers==2.4.0',
    'soupsieve==2.3.2.post1',
    'Sphinx==5.0.2',
    'sphinx-autoapi==2.0.1',
    'sphinx-basic-ng==1.0.0b1',
    'sphinx-copybutton==0.5.1',
    'sphinxcontrib-applehelp==1.0.2',
    'sphinxcontrib-devhelp==1.0.2',
    'sphinxcontrib-htmlhelp==2.0.0',
    'sphinxcontrib-jsmath==1.0.1',
    'sphinxcontrib-qthelp==1.0.3',
    'sphinxcontrib-serializinghtml==1.1.5',
    'stack-data==0.3.0',
    'statsmodels==0.13.2',
    'stringcase==1.2.0',
    'tables==3.7.0',
    'tabulate==0.8.10',
    'tenacity==8.1.0',
    'terminado==0.15.0',
    'threadpoolctl==3.1.0',
    'thriftpy2==0.4.14',
    'tinycss2==1.1.1',
    'tomli==2.0.1',
    'toolz==0.12.0',
    'torch==1.13.1',
    'torchmetrics==0.10.0',
    'torchsummary==1.5.1',
    'torchvision==0.14.1',
    'tornado==6.1',
    'tqdm==4.64.0',
    'traces==0.6.0',
    'traitlets==5.3.0',
    'twine',
    'typepy==1.3.0',
    'typing_extensions==4.2.0',
    'Unidecode==1.3.6',
    'urllib3==1.26.9',
    'validators==0.20.0',
    'vega-datasets==0.9.0',
    'wcwidth==0.2.5',
    'webencodings==0.5.1',
    'wfdb==3.4.1',
    'wget==3.2',
    'widgetsnbextension==3.6.1',
    'wrapt==1.14.1',
    'xarray==2023.12.0',
    'xmltodict==0.13.0',
    'zipp==3.8.1',
    'zope.interface==6.0'
]

setup(
    name='LongTermBiosignals',
    version='2.0.2',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=requirements,
    url='https://github.com/jomy-kk/LTBio',
    license='',
    author='João Saraiva, Mariana Abreu',
    author_email='joaomiguelsaraiva@tecnico.ulisboa.pt',
    description='Python library for easy managing and processing of large Long-Term Biosignals.',
    long_description = long_description,
    long_description_content_type = "text/markdown",

    python_requires = ">=3.10.4",

)
