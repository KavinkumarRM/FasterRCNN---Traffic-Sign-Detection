#! /bin/bash

dir=${0%/*}
if [ "$dir" = "$0" ]; then
  dir="."
fi
cd "$dir"
python video_local.py -p ./video.avi


