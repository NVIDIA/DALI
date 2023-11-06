#!/bin/bash
docker run -it -v $PWD:/workdir ghcr.io/streetsidesoftware/cspell:latest --words-only --unique --quiet "**" | sort --ignore-case >> project-words.txt