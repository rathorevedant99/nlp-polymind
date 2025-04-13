#!/bin/bash

counter=1
while [ $counter -le 10 ]; do
    python main.py
    echo "Run $counter completed"
    ((counter++))
done

echo "All runs completed"
