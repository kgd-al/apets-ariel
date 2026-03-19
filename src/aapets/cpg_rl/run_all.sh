#!/bin/bash

# Fail as quick as possible
set -euo pipefail

exp=$1
seeds=$2
shift 2

source $HOME/venv/bin/activate
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
max=${MAX_CONCURRENT:-20}

envs=${ENVS:-ariel gym}
trainers=${TRAINERS:-cma ppo}

prefix(){
  printf "[%s] " "$(date)"
}

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
  while [ $(squeue -u kgd | wc -l) -gt $max ]
  do
    prefix
    printf "Waiting for some room in queue\r"
    sleep 10
  done

  job_name=$exp/$env/$trainer/$reward/$name
  job_path=$(cut -d/ -f 2- <<< "$job_name")
  data_parent_folder=$data_root/$job_path

  slurm_logs_base="$slurm_logs/$job_path/"

  local_seeds=()
  for seed in $expanded_seeds
  do
    [ ! -d $data_parent_folder/run-$seed ] && local_seeds+=($seed)
  done
  local_seeds=$(IFS=,; echo "${local_seeds[*]}")

  prefix

  if [ -z "$local_seeds" ]
  then
    echo "Skipping $job_name, seeds $seeds already covered"
    continue
  fi

  sbatch -o "$slurm_logs_base/run-%a.out" -e "$slurm_logs_base/run-%a.err" <<EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$threads
#SBATCH --mem=7G
#SBATCH --array=$local_seeds
#SBATCH --time=$duration

seed=\$SLURM_ARRAY_TASK_ID
data_folder=$data_parent_folder/run-\$seed

source $HOME/venv/bin/activate

date
echo "Seed is \$seed"
echo "Saving data to \$data_folder"
echo "Additional arguments: $args $@"

export MUJOCO_GL=egl
(

  # Print set -x to stdout
  BASH_XTRACEFD=1

  set -x
  python -m aapets.cpg_rl.main --seed \$seed \
  --env $env --trainer $trainer --arch $arch --reward $reward \
  $args \
  --overwrite --budget $budget --duration 10 --threads $threads --data-folder \$data_folder
)

for ext in out err
do
  mv -v $slurm_logs_base/run-\$seed.\$ext \$data_folder/slurm.\$ext
done

rmdir -p --ignore-fail-on-non-empty $slurm_logs

EOF
  echo " with seeds $local_seeds"

  sleep 1
done
