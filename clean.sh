#!/usr/bin/env bash

echo "Removing assets"
for path in {"rave/bounces/*.wav","rave/plots/*.{jpg,png}","rave/csd/*.csd"}; do
    echo "🗑  deleting $path"
    rm -rf $path
done

