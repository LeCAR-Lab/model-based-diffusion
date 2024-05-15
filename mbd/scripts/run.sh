for env_name in "hopper" "halfcheetah" "ant" "walker2d" "humanoidstandup" "humanoidrun" "pushT" 
do
    echo "Running $env_name"
    python run.py --env_name $env_name --algo path_integral
done