import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
	   name='quickemcee',
	   packages=['quickemcee'],
	   version='0.0.1a5',
	   description=('quickemcee is a library with built in scripts'
		        'using the emcee library to quickly set up '
			'MCMC analysis.'),
	   long_description=long_description,
	   long_description_content_type="text/markdown",
	   url='https://github.com/sofia-scz/quick-emcee/',
	   author='Sofia A. Scozziero',
	   author_email='sgscozziero@gmail.com',
	   install_requires=['numpy','scipy','matplotlib','emcee','corner'], 
	   classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: POSIX :: Linux",]
)
