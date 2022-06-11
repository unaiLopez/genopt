from setuptools import setup, find_packages

setup(
    name='genetist',
    version='0.1',
    license='MIT',
    author='Unai Lopez Ansoleaga',
    author_email='unai19970315@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/unaiLopez/genetist',
    keywords='genetic algorithm optimization',   
    install_requires=[
          'numpy', 'pandas', 'tqdm', 'logging'
      ],

)