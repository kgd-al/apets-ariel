#!/bin/bash

# Fail as quick as possible
set -euo pipefail

exp=$1
seeds=$2
shift 2

[[ $(uname -a ) =~ "vetinari" ]] && source $HOME/venv/bin/activate

expanded_seeds=$(python <<EOF

seeds=set()
for token in "$seeds".split(","):
  if "-" in token:
    l, u = token.split("-")
    seeds.update(list(range(int(l), int(u)+1)))
  else:
    seeds.add(int(token))
print(" ".join([str(s) for s in sorted(seeds)]))

EOF
)
echo "Expanded seed set: $expanded_seeds"

data_root=$HOME/data/$exp
mkdir -p "$data_root"

slurm_logs=$data_root/_slurm_logs/
mkdir -p "$slurm_logs"

budget=${BUDGET:-2_000_000}
threads=${THREADS:-8}
duration=${SLURM_DURATION:-10:00:00}
partition=${SLURM_PARTITION:-batch}

#envs=${ENVS:-ariel gym}
envs=${ENVS:-ariel}
trainers=${TRAINERS:-cma ppo}

prefix(){
  printf "[%s] " "$(date)"
}

jobs=.jobs.$(date +%s).slurm_array

(
  for env in $envs
  do
    for reward in speed gym kernels
    do
      for trainer in cma
      do
        for neighborhood in 0 2 4 6
        do
          echo $env cma cpg $reward cpg-$neighborhood --cpg-neighborhood $neighborhood
        done
      done

      for trainer in $trainers
      do
        echo $env $trainer mlp $reward mlp-0-0 --mlp-width 0 --mlp-depth 0
        for width in 1 2 4 8 16 32 64 128
        do
          for depth in 1 2
          do
            echo $env $trainer mlp $reward mlp-$depth-$width --mlp-width $width --mlp-depth $depth
          done
        done
      done
    done
  done
) | grep -v -e "cma.*--mlp-width 64 --mlp-depth 2" -e "cma.*--mlp-width 128" \
  | while read env trainer arch reward name args
do
  for seed in $expanded_seeds
  do
    job_name=$exp/$env/$trainer/$reward/$name/run-$seed
    job_path=$(cut -d/ -f 2- <<< "$job_name")
    data_folder=$data_root/$job_path

    [ -d $data_folder ] && continue
#    echo $job_name $data_folder
    echo $data_folder \
      python -m aapets.cpg_rl.main --seed $seed \
        --env $env --trainer $trainer --arch $arch --reward $reward \
        $args \
        --no-overwrite --budget $budget --duration 10 --threads $threads --data-folder $data_folder
  done
done | nl -v0 -w1 -s ' ' > $jobs

njobs=$(wc -l < $jobs)
array=0-$((njobs-1))
echo "Scheduling n=$njobs jobs (array=$array)"

sbatch -o "$slurm_logs/%x-%a.out" -e "$slurm_logs/%x-%a.err" <<EOF
#!/bin/bash

#SBATCH --job-name=$exp
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$threads
#SBATCH --mem=7G
#SBATCH --array=$array
#SBATCH --time=$duration

task_id=\$SLURM_ARRAY_TASK_ID

line=\$(grep "^\$task_id " $jobs)
if [ -z "\$line" ]
then
  echo "Failed at grabbing line \$task_id from $jobs"
else
  echo "Grabbing line \$task_id from $jobs: '\$line'"
fi
read id folder cmd <<< "\$line"

source $HOME/venv/bin/activate

date
echo "Saving data to \$folder"

export MUJOCO_GL=egl
(

  # Print set -x to stdout
  BASH_XTRACEFD=1

  set -x
  \$cmd
)

for ext in out err
do
  mv -v $slurm_logs/$exp-\$task_id.\$ext \$folder/slurm.\$ext
done

rmdir -p --ignore-fail-on-non-empty $slurm_logs

EOF
