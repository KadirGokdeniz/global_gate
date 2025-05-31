#!/bin/sh
until docker inspect -f '{{.State.Status}}' global-gate-scraper | grep -q 'exited'; do
  sleep 1;
done;
exit $(docker inspect -f '{{.State.ExitCode}}' global-gate-scraper)