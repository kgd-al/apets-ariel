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

tmp_root=/tmp/$USER/$exp

slurm_logs=$data_root/_slurm_logs/
mkdir -p "$slurm_logs"

population=${POPULATION:-100}
generations=${GENERATIONS:-100}
learning=${LEARNING:-100}
threads=${THREADS:-8}
duration=${SLURM_DURATION:-24:00:00}
partition=${SLURM_PARTITION:-batch}
limits=${LIMITS:-}

tasks=${TASKS:-locomotion}
symmetries=${SYMMETRIES:-none body both}

if [[ -n $limits ]]
then
  limits="%$limits"
fi

echo "  Experiment: $exp"
echo " Data Folder: $data_root"
echo "  Tmp Folder: $tmp_root"
echo "     Threads: $threads"
echo "    Duration: $duration"
echo "   Partition: $partition"
echo "  Population: $population"
echo " Generations: $generations"
echo "    Learning: $learning"
echo "       Tasks: $tasks"
echo "  Symmetries: $symmetries"

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
      echo $task $symmetry
    done
  done
) | while read task symmetry args
do
  for seed in $expanded_seeds
  do
    job_name=$task/$symmetry/run-$seed
    data_folder=$data_root/$job_name
    tmp_folder=$tmp_root/$job_name

    [ -d $data_folder ] && continue
#    echo $job_name $data_folder >&2
    echo $data_folder $tmp_folder \
      python -m aapets.g_cpg.main --seed $seed \
        --task $task --symmetry $symmetry \
        $args \
        --no-overwrite --threads $threads --data-folder $tmp_folder \
        --population-size $population --generations $generations --learning $learning
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
#SBATCH --mem=5G
#SBATCH --array=$array$limits
#SBATCH --time=$duration
#SBATCH --exclude=node01

# To maybe get a GPU when rendering. Bad idea!
####SBATCH --gres=gpu:1

task_id=\$SLURM_ARRAY_TASK_ID

line=\$(grep "^\$task_id " $jobs)
if [ -z "\$line" ]
then
  echo "Failed at grabbing line \$task_id from $jobs"
else
  echo "Grabbing line \$task_id from $jobs: '\$line'"
fi
read id data_folder tmp_folder cmd <<< "\$line"

finalize(){
  rc=\$?
  data_parent=\$(dirname \$data_folder)
  mkdir -p \$data_parent

  mv -v \$tmp_folder \$data_parent

  for ext in out err
  do
    mv -v $slurm_logs/$exp-\$task_id.\$ext \$data_folder/slurm.\$ext
  done

  exit \$rc
}
trap finalize EXIT

source $HOME/venv/bin/activate

date
echo "Saving data to \$tmp_folder (later to be moved to \$data_folder)"

export MUJOCO_GL=egl
(

  # Print set -x to stdout
  BASH_XTRACEFD=1

  set -x
  \$cmd
)

rmdir -p --ignore-fail-on-non-empty $slurm_logs

EOF
