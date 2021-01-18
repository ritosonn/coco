#!/bin/bash
cd `dirname $0`
docker run --gpus all -it --rm --net=host --shm-size 4g -v `dirname $(pwd)`:/coco/ coco bash 