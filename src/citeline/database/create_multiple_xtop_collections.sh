#!/usr/bin/env bash

n=512

while [ $n -ge 1 ]; do
    ./create_xtop.sh $n
    n=$((n / 2))
done