set -e
pip install external/fv3gfs-util
pip install jinja2 cython
make -C external/fv3gfs-wrapper/lib
pip install -e external/fv3gfs-wrapper
pip install fv3config
