from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

INSTALL_REQUIRES = [
    'numpy==1.21.3',
    'tqdm==4.64.0',
    'pandas==1.3.4',
]

if __name__ == '__main__':
    setup(
        name='genetist',
        description='Genetist: optimization with genetic algorithms',
        long_description=long_description,
        version=0.2,
        long_description_content_type='text/markdown',
        author='Unai Lopez Ansoleaga',
        author_email='unai19970315@gmail.com',
        url='https://github.com/unaiLopez/genetist',
        license='MIT',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        python_requires='>=3.6'
    )