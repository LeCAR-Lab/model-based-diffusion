parallel_num=8
total_frames=1500
frame_per_thread=$((total_frames / parallel_num))

# record time
start_time=$(date +%s)

# plot
for i in $(seq 0 $((parallel_num - 1))); do
    start_frame=$((i * frame_per_thread))
    end_frame=$((start_frame + frame_per_thread))
    if [ $i -eq $((parallel_num - 1)) ]; then
        end_frame=$total_frames
    fi
    python devel/plot_npy.py --start $start_frame --end $end_frame &
done

# wait for all threads to finish
wait

# render video
rm figure/render.mp4
ffmpeg -framerate 30 -i figure/%d.png -c:v libx264 -pix_fmt yuv420p figure/render.mp4
rm figure/*.png

# record time
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time)) seconds"