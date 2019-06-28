from setuptools import setup, find_packages

setup(name='text.pt',
      version='1.0',
      description='Helper classes to manipulate text corpora in pytorch models',
      author='Bryan Eikema and Wilker Aziz',
      author_email='b.eikema@uva.nl',
      url='https://github.com/probabll/text.pt',
      packages=find_packages(),
      include_package_data=True
)
