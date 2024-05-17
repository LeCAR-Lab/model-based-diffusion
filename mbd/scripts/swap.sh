cd ../planners

env_name="humanoidstandup"
echo "Swapping MBD temperature for $env_name"
# swap tempertures from 1.0 to 0.1 in 0.1 increments
for temp in $(seq 1.0 -0.1 0.1)
do
    echo "temperature = $temp"
    python mc_mbd.py --temp_sample $temp --silent --env_name $env_name --disable_recommended_params
done