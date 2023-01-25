#!/bin/bash

function csdiffv3 ()
{
   cp ./left.py /home/ze/custom-separators-merge-tool/temp/left.py
   cp ./right.py /home/ze/custom-separators-merge-tool/temp/right.py
   cp ./base.py /home/ze/custom-separators-merge-tool/temp/base.py
   cd /home/custom-separators-merge-tool/

   bash update_retult.sh
   cd -

   cp  /home/ze/custom-separators-merge-tool/temp/csdiff.py ./csdiff.py
}

