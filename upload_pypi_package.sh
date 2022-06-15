#!/bin/sh
rm -r build/
rm -r dist/
rm -r genetist.egg-info/

python setup.py bdist_wheel --universal
python setup.py sdist
twine upload dist/* -u $1 -p $2