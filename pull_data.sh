#!/bin/sh

rsync -az tw@timakro.de:/srv/tmlp/sessions/ sessions
./analyze_data.sh
