#!/bin/bash

# install the needed fv3net packages
for package in /external/*
do
    echo "Setting up $package"
    pip install -e "$package" --no-deps > /dev/null 2> /dev/null
done

exec "$@"