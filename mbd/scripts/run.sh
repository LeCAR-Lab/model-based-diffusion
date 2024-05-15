for env_name in "hopper" "halfcheetah" "ant" "walker2d" "humanoidstandup" "humanoidrun"; do #"pushT"
    for update_method in "softmax" "cma-es" "cem"; do
        echo "Running $env_name with $update_method"
        python run.py --env_name $env_name --algo path_integral --update_method $update_method
    done
done