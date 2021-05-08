source $stdenv/setup

if [[ $src  == *.tar.gz ]]; then 
    echo "Unpacking tar file $src"
    mkdir src
    tar -C src --strip-components=1 -xzf $src
elif [[ -d $src ]]; then
    echo "Directory detected"
    cp -r $src src
    chmod -R +w src/
fi

echo "Building sdist"
cd src/
python setup.py sdist
mv dist/*.tar.gz $out