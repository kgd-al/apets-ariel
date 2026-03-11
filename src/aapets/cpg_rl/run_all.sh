#!/bin/bash

name=$1
seeds=$2
shift 2

data_root=$HOME/data/$name
mkdir -p "$data_root"

slurm_logs=$data_root/slurm_logs/$name/
mkdir -p "$slurm_logs"

job_name=$name

slurm_logs_base="$slurm_logs/run-%a"

duration=${SLURM_DURATION:-10:00:00}
threads=${THREADS:-8}
partition=${SLURM_PARTITION:-batch}

run_one(){
  env=$1
  trainer=$2
  arch=$3
  reward=$4
  name=$5
  shift 5
  echo python -m aapets.cpg_rl.main --seed 0 \
    --env $env --trainer $trainer --arch $arch --reward $reward \
    $@ \
    --overwrite --budget 200 --duration 1 --threads 1 --data-folder $HOME/data/cpg_rl/$env/$trainer/$reward/$name/run-0/
}

prefix(){
  printf "[%s] " "$(date)"
}

(
  for env in ariel gym
  do
    for reward in speed gym kernels
    do
      for neighborhood in 0 2 4 6
      do
        run_one $env cma cpg $reward cpg-$neighborhood --cpg-neighborhood $neighborhood
      done

      for trainer in cma ppo
      do
        run_one $env $trainer mlp $reward mlp-0-0 --mlp-width 0 --mlp-depth 0 $i
        for width in 1 2 4 8 16 32 64 128
        do
          for depth in 1 2
          do
            run_one $env $trainer mlp $reward mlp-$depth-$width --mlp-width $width --mlp-depth $depth $i
          done
        done
      done
    done
  done
) | grep -v -e "--trainer cma.*--mlp-width 64 --mlp-depth 2" -e "--trainer cma.*--mlp-width 128" | while read task
do
  while [ $(squeue -u kgd | wc -l) -gt 20 ]
  do
    prefix
    printf "Waiting for some room in queue\r"
    sleep 10
  done

  prefix
  echo sbatch -o "$slurm_logs_base.out" -e "$slurm_logs_base.err" <<EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$threads
#SBATCH --array=$seeds
#SBATCH --time=$duration

seed=\$SLURM_ARRAY_TASK_ID
data_folder=$HOME/data/cpg_rl/$env/$trainer/$reward/$name//run-\$seed

source $HOME/venv/bin/activate

date
echo "Seed is \$seed"
echo "Saving data to \$data_folder"
echo "Additional arguments: $@"

export MUJOCO_GL=egl
(
  set -x
  $cmd --data_folder $data_folder
)

for ext in out err
do
  mv -v $slurm_logs/run-\$seed.\$ext \$data_folder/slurm.\$ext
done

rmdir -p --ignore-fail-on-non-empty $slurm_logs

EOF
  sleep 1
done
