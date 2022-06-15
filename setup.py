from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

INSTALL_REQUIRES = [
    'numpy==1.21.3',
    'tqdm==4.64.0',
    'pandas==1.3.4',
]

VERSION = '0.9.9'

if __name__ == '__main__':
    setup(
        name='genetist',
        description='Genetist: optimization with genetic algorithms',
        long_description=long_description,
        version=VERSION,
        long_description_content_type='text/markdown',
        author='Unai Lopez Ansoleaga',
        author_email='unai19970315@gmail.com',
        url='https://github.com/unaiLopez/genetist',
        license='MIT',
        packages=['genetist'],
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        python_requires='>=3.6'
    )
