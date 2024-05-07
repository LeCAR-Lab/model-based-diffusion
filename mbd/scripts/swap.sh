cd ../planners

env_name="halfcheetah"
echo "Running MBD on $env_name"
# swap tempertures from 1.0 to 0.1 in 0.1 increments
for temp in $(seq 1.0 -0.1 0.1)
do
    echo "Swapping temperature to $temp"
    python mc_mbd.py --temp_sample $temp --silent --env_name $env_name
done