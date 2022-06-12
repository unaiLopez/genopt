from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

INSTALL_REQUIRES = [
    'numpy==1.22',
    'tqdm==4.4.1',
    'pandas==1.1.5',
]

if __name__ == '__main__':
    setup(
        name='darwin',
        description='darwin: optimization with genetic algorithms',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Unai Lopez Ansoleaga',
        author_email='unai19970315@gmail.com',
        uurl='https://github.com/unaiLopez/darwin',
        license='MIT',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        python_requires='>=3.6',
    )