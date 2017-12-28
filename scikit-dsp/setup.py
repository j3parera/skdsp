#!/usr/bin/env python

import os
import subprocess
import sys
from setuptools import setup, find_packages


descr = """\
A library for DSP
"""
DISTNAME = 'scikit-dsp'
DESCRIPTION = 'A library for DSP'
LONG_DESCRIPTION = descr
MAINTAINER = 'José Parera',
MAINTAINER_EMAIL = 'jose.parera@upm.es',
URL = 'http://projects.scipy.org/scipy/scikits'
LICENSE = 'MIT'
DOWNLOAD_URL = URL
PACKAGE_NAME = 'skdsp'
EXTRA_INFO = dict(
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    # install_requires=['numpy', 'sympy'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: Free for non-commercial use',
                 'Topic :: Scientific/Engineering']
)


def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg: "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(PACKAGE_NAME)
    return config


def get_version():
    """Obtain the version number"""
    import imp
    mod = imp.load_source('version', os.path.join(PACKAGE_NAME, 'version.py'))
    return mod.__version__


# Documentation building command
try:
    from sphinx.setup_command import BuildDoc as SphinxBuildDoc

    class BuildDoc(SphinxBuildDoc):
        """Run in-place build before Sphinx doc build"""
        def run(self):
            ret = subprocess.call([sys.executable, sys.argv[0],
                                   'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building Scipy failed!")
            SphinxBuildDoc.run(self)
    cmdclass = {'build_sphinx': BuildDoc}
except ImportError:
    cmdclass = {}

# Call the setup function
if __name__ == "__main__":
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          include_package_data=True,
          test_suite="nose.collector",
          cmdclass=cmdclass,
          version=get_version(),
          **EXTRA_INFO)
