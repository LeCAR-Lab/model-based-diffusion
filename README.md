# diffuser-planer

1. dyn_scale = 1.0 ✅
2. variance schedule depends on the barrier
3. add visualization ✅

1. change initialization (make the angle orient to outside)
2. change rotation action scale
3. change scale update rule (initially, only consider dynamics)

limitation
1. still have to tune different objective update rules. example: if final loss is too high, will make later points stuck, if barrier goes up too fast, will break the dynamics, for the reward term, the update rule based on the dynamics is not good (for instance, if increase the reward term too fast, it will converge early)
2. what about also add initial state into the reward term?