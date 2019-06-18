#!/bin/bash

set -e

docker run -d --name gitlab-runner --restart always \
   -v /srv/gitlab-runner/config:/etc/gitlab-runner \
   -v /var/run/docker.sock:/var/run/docker.sock \
   gitlab/gitlab-runner:latest

docker run -v /srv/gitlab-runner/config:/etc/gitlab-runner gitlab/gitlab-runner register \
  --non-interactive \
  --url "https://gitlab-master.nvidia.com/" \
  --registration-token "mH3YvhoxzGtVecvxesyx" \
  --executor "docker" \
  --docker-image dali-docker \
  --docker-volumes /var/run/docker.sock:/var/run/docker.sock \
  --description "mszolucha-trtx" \
  --tag-list "mszolucha" \
  --run-untagged="false" \
  --locked="true"
