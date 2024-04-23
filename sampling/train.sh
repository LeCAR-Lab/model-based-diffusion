# loop backend over spring, positional, generalized
for backend in spring positional generalized; do
    for env_name in halfcheetah hopper walker2d; do
        echo "Training $env_name with $backend backend"
        python train_brax.py --env_name $env_name --backend $backend
    done
done
