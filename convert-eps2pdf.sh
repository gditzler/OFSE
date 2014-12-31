#!/usr/bin/env bash 

files=`find eps/*.eps`

for file in ${files[@]}; do 
  echo "Converting $file"
  epstopdf $file
done
mv eps/*.pdf pdf/
