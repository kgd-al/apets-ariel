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

budget=${BUDGET:-2_000_000}
threads=${THREADS:-8}
duration=${SLURM_DURATION:-10:00:00}
partition=${SLURM_PARTITION:-batch}
max=${MAX_CONCURRENT:-20}

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
        echo $env cma cpg $reward cpg-$neighborhood --cpg-neighborhood $neighborhood
      done

      for trainer in cma ppo
      do
        echo $env $trainer mlp $reward mlp-0-0 --mlp-width 0 --mlp-depth 0 $i
        for width in 1 2 4 8 16 32 64 128
        do
          for depth in 1 2
          do
            echo $env $trainer mlp $reward mlp-$depth-$width --mlp-width $width --mlp-depth $depth $i
          done
        done
      done
    done
  done
) | grep -v -e "--trainer cma.*--mlp-width 64 --mlp-depth 2" -e "--trainer cma.*--mlp-width 128" \
  | while read env trainer arch reward name args
do
  while [ $(squeue -u kgd | wc -l) -gt max ]
  do
    prefix
    printf "Waiting for some room in queue\r"
    sleep 10
  done

  data_parent_folder=$HOME/data/cpg_rl/$env/$trainer/$reward/$name/

  prefix
  sbatch -o "$slurm_logs_base.out" -e "$slurm_logs_base.err" <<EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$threads
#SBATCH --array=$seeds
#SBATCH --time=$duration

seed=\$SLURM_ARRAY_TASK_ID
data_folder=$data_parent_folder/run-\$seed

echo python -m aapets.cpg_rl.main --seed \$seed \
  --env $env --trainer $trainer --arch $arch --reward $reward \
  $args \
  --overwrite --budget $budget --duration 15 --threads $threads --data-folder \$data_folder

source $HOME/venv/bin/activate

date
echo "Seed is \$seed"
echo "Saving data to \$data_folder"
echo "Additional arguments: $args $@"

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
