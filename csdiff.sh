#!/bin/bash

sed 's/{/\n$$$$$$${\n$$$$$$$/g' $1 | sed 's/}/\n$$$$$$$}\n$$$$$$$/g' | sed 's/(/\n$$$$$$$(\n$$$$$$$/g' | sed 's/)/\n$$$$$$$)\n$$$$$$$/g' | sed 's/,/\n$$$$$$$,\n$$$$$$$/g' > "temp_$1"
sed 's/{/\n$$$$$$${\n$$$$$$$/g' $2 | sed 's/}/\n$$$$$$$}\n$$$$$$$/g' | sed 's/(/\n$$$$$$$(\n$$$$$$$/g' | sed 's/)/\n$$$$$$$)\n$$$$$$$/g' | sed 's/,/\n$$$$$$$,\n$$$$$$$/g' > "temp_$2"
sed 's/{/\n$$$$$$${\n$$$$$$$/g' $3 | sed 's/}/\n$$$$$$$}\n$$$$$$$/g' | sed 's/(/\n$$$$$$$(\n$$$$$$$/g' | sed 's/)/\n$$$$$$$)\n$$$$$$$/g' | sed 's/,/\n$$$$$$$,\n$$$$$$$/g' > "temp_$3"

diff3 -m "temp_$1" "temp_$2" "temp_$3" > mid_merged

rm "temp_$1"
rm "temp_$2"
rm "temp_$3"

sed ':a;N;$!ba;s/\n\$\$\$\$\$\$\$//g' mid_merged > merged

rm mid_merged

sed "s/temp_$1/$1/g" merged | sed "s/temp_$2/$2/g" | sed "s/temp_$3/$3/g" | sed "s/\(<<<<<<< $1\)\(.\+\)/\1\n\2/" | sed "s/\(||||||| $2\)\(.\+\)/\1\n\2/" | sed "s/\(=======\)\(.\+\)/\1\n\2/" | sed "s/\(>>>>>>> $3\)\(.\+\)/\1\n\2/"

 rm merged


