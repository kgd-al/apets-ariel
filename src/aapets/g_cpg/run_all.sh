#!/bin/bash

# Fail as quick as possible
set -euo pipefail

name=g_cpg
exp=$1
seeds=$2
shift 2

log(){
  echo "[$(date)] $@"
}

log "Preparing jobs list for exp=$exp and seeds=$seeds"

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

data_root=$HOME/data/$name/$exp
mkdir -p "$data_root"

slurm_logs=$data_root/_slurm_logs/
mkdir -p "$slurm_logs"

population=${POPULATION:-100}
generations=${GENERATIONS:-100}
learning=${LEARNING:-100}
threads=${THREADS:-8}
duration=${SLURM_DURATION:-48:00:00}
partition=${SLURM_PARTITION:-batch}
mem_limit=${MEMORY:-20}
limits=${LIMITS:-}

tasks=${TASKS:-locomotion compliance}
symmetries=${SYMMETRIES:-none body both}

morphologies=${MORPHOLOGIES:-spider ariel_ant gym_ant unitree_go1}

if [[ -n $limits ]]
then
  limits="%$limits"
fi

echo "   Experiment: $exp"
echo "       Folder: $data_root"
echo "      Threads: $threads"
echo "     Duration: $duration"
echo "    Partition: $partition"
echo "       Memory: $mem_limit (Gb)"
echo "   Population: $population"
echo "  Generations: $generations"
echo "     Learning: $learning"
echo "        Tasks: $tasks"
echo "   Symmetries: $symmetries"
echo " Morphologies: $morphologies"

read -p "All good? [Yy]es " -n 1 -r go
[[ "$go" =~ ^[Yy]$ ]] || (echo; exit 2)
echo

jobs=.jobs.$name.$(date +%s).slurm_array
# rm -f .jobs.$name.*.slurm_array

(
  for task in $tasks
  do
    for symmetry in $symmetries
    do
      echo evo/$task/$symmetry $task --symmetry $symmetry
    done
    for body in $morphologies
    do
      echo fixed/$task/$body $task --fixed-morphology $body 
    done
  done
) | while read folder task args
do
  for seed in $expanded_seeds
  do
    data_folder=$data_root/$folder/run-$seed

    [ -d $data_folder ] && continue
    echo $data_folder \
      python -m aapets.g_cpg.main --seed $seed \
        --task $task  \
        --no-overwrite --threads $threads --data-folder $data_folder \
        --population-size $population --generations $generations --learning $learning \
        --duration 10 $args \

  done
done | nl -v0 -w1 -s ' ' > $jobs

njobs=$(wc -l < $jobs)
array=0-$((njobs-1))
log "Scheduling n=$njobs jobs (array=$array)"

if [ $njobs -eq 0 ]
then
  echo
  echo "All folders exist for requested jobs. Nothing to do"
  exit 0
fi

sbatch -o "$slurm_logs/%x-%a.out" -e "$slurm_logs/%x-%a.err" <<EOF
#!/bin/bash

#SBATCH --job-name=$exp
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$threads
#SBATCH --mem=${mem_limit}G
#SBATCH --array=$array$limits
#SBATCH --time=$duration
#SBATCH --exclude=node01

# To maybe get a GPU when rendering. Bad idea!
####SBATCH --gres=gpu:1

task_id=\$SLURM_ARRAY_TASK_ID

echo "Running job \$SLURM_JOB_ID.\$task_id"
line=\$(grep "^\$task_id " $jobs)
if [ -z "\$line" ]
then
  echo "Failed at grabbing line \$task_id from $jobs"
else
  echo "Grabbing line \$task_id from $jobs: '\$line'"
fi
read id folder cmd <<< "\$line"

finalize(){
  rc=\$?
  for ext in out err
  do
    mv -v $slurm_logs/$exp-\$task_id.\$ext \$folder/slurm.\$ext
  done

  exit \$rc
}
trap finalize EXIT

source $HOME/venv/bin/activate

date
echo "Saving data to \$data_folder"

mem_limit=\$(($mem_limit*1024*1024))
echo "Enforcing strict memory limits of \$mem_limit bytes ($mem_limit Gb)"
ulimit -v \$mem_limit

export MUJOCO_GL=egl
\$cmd

rmdir -p --ignore-fail-on-non-empty $slurm_logs

EOF
