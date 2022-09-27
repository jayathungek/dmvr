#!/bin/bash 

csv_file="$1"
out_dir="$2"
yt_prefix="https://www.youtube.com/watch?v="
total_vids=$(tail +4 "$csv_file" | wc -l ) # get rid of 3 header lines

seconds_to_timestamp() {
  #arg1 integer (seconds)
  secs="$1"
  printf '%d:%d:%d\n' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

dl_from_id() {
  #arg1: youtube ID #arg2: start time
  id="$1"
  url="$yt_prefix$id"
  start="$2"
  direct_url=$(yt-dlp --youtube-skip-dash-manifest -g -f18 "$url" 2>/dev/null)
  # video_url=$(echo "$video_audio_urls" | grep 'mime=video')
  # audio_url=$(echo "$video_audio_urls" | grep 'mime=audio')
  if [[ -z "$direct_url"  ]]; then
    return 2 
  fi
  
  if [ ! -f "$out_dir/$id.webm" ]; then 
    ffmpeg -hide_banner -loglevel error -nostdin -ss "$start" -t 10 -i "$direct_url" -c:v libx264 -preset slow -crf 22 "$out_dir/$id.mp4" &>/dev/null 
    # yt-dlp --postprocessor-args "-ss $start -t 10" "$url"  -o "$out_dir/$id.webm" &>/dev/null
  fi

}

curr_vid=1
failed_vids=0
while read -r line; do
  yt_id=$(echo "$line" | cut -d, -f1)
  start_time=$(echo "$line" | cut -d, -f2 | cut -d. -f1)
  start=$(seconds_to_timestamp "$start_time")
  end_ts=$(seconds_to_timestamp $((start_time + 10 )) )
  
  if [ ! "$curr_vid" -eq "$total_vids" ]; then
    echo -en "Downloading $yt_id from $start -> $end_ts [$curr_vid/$total_vids] [$failed_vids failed]\e[K\r"
  else
    echo -en "Downloading $yt_id from $start -> $end_ts [$curr_vid/$total_vids] [$failed_vids failed]\n"
  fi
  dl_from_id "$yt_id" "$start"
  err_code="$?"
  if [ ! "$err_code" -eq 0 ]; then
    failed_vids=$((failed_vids + 1 ))
    if [ "$err_code" -eq 1 ]; then
      echo "$yt_id [ffmpeg error]" >> "$out_dir/failed_vids.txt"
    elif [ "$err_code" -eq 2 ]; then
      echo "$yt_id [yt-dlp error]" >> "$out_dir/failed_vids.txt"
    else
      echo "$yt_id [unkown error]" >> "$out_dir/failed_vids.txt"
    fi
      
  fi
  curr_vid=$((curr_vid + 1 ))
done< <(tail +4 "$csv_file") # get rid of 3 header lines

