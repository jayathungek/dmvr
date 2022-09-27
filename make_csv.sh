#!/bin/bash

in_dir="$1"  # directory that contains raw videos
og_filelist="$2"  # Original filelist that contains all the labels 
outfile="$in_dir/filelist.csv"

header="video_path,start,end,label"
file_regex="*.mp4"

usage() {
  printf "usage:\n%s\n" \
  "./make_csv <in_dir> <og_filelist>" \
  "<in_dir>: Directory that contains raw videos" \
  "<og_filelist>: Original filelist that contains all the labels"
}

get_label() {
  #arg1 Video ID

  vid_id="$1"
  line=$(cat "$og_filelist" | grep -- "$vid_id")
  label=$(echo "$line" | rev | cut -d' ' -f1 | rev)
  echo "$label"
}

if [[ -z "$in_dir" || -z "$og_filelist" ]]; then
  usage 
  exit 1
fi

echo "$header" > "$outfile"
curr_file=1
total_files=$(ls "$in_dir" | grep -- mp4 | wc -l)
while read -r filename; do 
  if [[ $filename == $file_regex ]]; then
    id=$(echo "$filename"  | cut -d. -f1 )
    filepath=$(echo $(readlink -e -- "$in_dir/$filename"))
    label=$(get_label "$id") 
    echo "$filepath, 0, 10, $label" >> "$outfile"
  fi
  if [ ! "$curr_file" -eq "$total_files" ]; then
    echo -en "Processed $curr_file/$total_files files\e[K\r"
  else
    echo -en "Processed $curr_file/$total_files files\n"
  fi
  curr_file=$((curr_file + 1))
done < <(ls "$in_dir")
