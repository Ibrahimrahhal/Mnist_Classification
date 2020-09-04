#!/bin/bash
echo "Please choose the name (default main)"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

read envName
echo $envName
if [ -z $envName ] 
then
envName="main"
fi
path="${DIR}/../environments/${envName}.yml"
echo "Starting exporting to ${path}...."
conda env export > $path
echo "Exporting finished."

