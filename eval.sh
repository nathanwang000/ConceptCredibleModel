#!/bin/bash

echo "biased Edema"
for i in {1..10};
do
    echo $i
    python mimic_scripts/stdSonT.py --task Edema -s noise -t 0.95 --subsample gender --eval --name standard_$i -o outputs
done

echo "clean Edema"
for i in {1..10};
do
    python mimic_scripts/stdSonT.py --task Edema -s clean --eval --name standard_$i -o outputs
done

echo "clean is_male"
for i in {1..10};
do
    python mimic_scripts/stdSonT.py --task is_male -s clean --eval --name standard_$i -o outputs
done

