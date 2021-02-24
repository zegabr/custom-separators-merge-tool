#!/bin/bash

sed 's/{/\n$${\n$$/g' $1 | sed 's/}/\n$$}\n$$/g' > "temp_$1"
sed 's/{/\n$${\n$$/g' $2 | sed 's/}/\n$$}\n$$/g' > "temp_$2"
sed 's/{/\n$${\n$$/g' $3 | sed 's/}/\n$$}\n$$/g' > "temp_$3"

diff3 -m "temp_$1" "temp_$2" "temp_$3" > mid_merged

rm "temp_$1"
rm "temp_$2"
rm "temp_$3"

process_merge_code="
import sys\n\nfile_to_open = sys.argv[1]\nmerged_file_to_write = sys.argv[2]\n\nmerged_file = open(file_to_open, 'r')\n\nlines = merged_file.readlines()\nfile_s = ''.join(lines)\n\nfile_ret = file_s.replace('\\\\n\$\$', '')\n\nfile2 = open(merged_file_to_write, 'w')\nfile2.write(file_ret)\nfile2.close()"

echo $process_merge_code > process_merge_code.py

python3 process_merge_code.py mid_merged merged

rm process_merge_code.py

rm mid_merged

sed "s/temp_$1/$1/g" merged | sed "s/temp_$2/$2/g" | sed "s/temp_$3/$3/g"

rm merged


