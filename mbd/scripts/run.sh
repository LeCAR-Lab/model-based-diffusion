for env_name in "pushT"; do #"hopper" "halfcheetah" "ant" "walker2d" "humanoidstandup" "humanoidrun" "pushT"; do
    for update_method in "softmax" "cma-es" "cem"; do
        echo "Running $env_name with $update_method"
        python run.py --env_name $env_name --algo path_integral --update_method $update_method
    done
    #echo "Running $env_name with mbd"
    #python run.py --env_name $env_name --algo mc_mbd
done

# for env_name in "hopper" "halfcheetah" "ant" "walker2d" "humanoidstandup" "humanoidrun" "pushT"; do
#     python train_brax.py --env_name $env_name 
# done
