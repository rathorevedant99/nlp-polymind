#!/bin/bash

counter=1
while [ $counter -le 30 ]; do
    python num-experts-ablation.py
    echo "Run $counter completed"
    ((counter++))
done

echo "All runs completed"
