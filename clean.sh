#!/usr/bin/env bash

BOUNCES="rave/bounces/*.wav"
PLOTS="rave/plots/*.{jpg,png}"
echo "Removing plots and bounces"
for path in {$BOUNCES,$PLOTS}; do
    echo "🗑  deleting $path"
    rm -rf $path
done

