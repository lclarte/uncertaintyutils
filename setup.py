from setuptools import setup, find_packages

VERSION = '1.0' 

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="uncertaintyutils", 
        version=VERSION,
        packages=["uncertaintyutils", "uncertaintyutils.data", "uncertaintyutils.gamp", "uncertaintyutils.erm", 
                "uncertaintyutils.gamp.prior", "uncertaintyutils.gamp.likelihood"], # this must be the same as the name above
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
)