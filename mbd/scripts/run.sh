for env_name in "pushT" "halfcheetah" "walker2d" "humanoidstandup" "humanoidrun"
do
    echo "Running $env_name"
    python train_brax.py --env_name $env_name
done