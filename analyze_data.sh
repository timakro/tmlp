#!/bin/sh

ls sessions/ | wc -l
find sessions/ -name meta.json|xargs grep volume|cut -d' ' -f 3|awk '{s+=$1} END {print s}'
du -sh sessions
