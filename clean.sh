#!/usr/bin/env bash

echo "Removing assets"
for path in {"rave/bounces/*.wav","rave/plots/*.{jpg,png}","rave/csd/*.csd"}; do
    echo "🗑  deleting $path"
    rm -rf $path
done

function delete_logs() {
    echo "Removing logs"
    rm -rf rave/ray_results/*
}


if [ "$1" == "--force" -o "$1" == "-f" ]; then
        delete_logs;
else
        read -p "This will delete all the ray logs. Are you sure? (y/n) " -n 1;
        echo "";
        if [[ $REPLY =~ ^[Yy]$ ]]; then
                delete_logs;
        fi;
fi;
unset delete_logs;
