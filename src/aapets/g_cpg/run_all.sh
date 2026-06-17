#!/bin/bash

# Fail as quick as possible
set -euo pipefail

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

data_root=$HOME/data/g_cpg/$exp
mkdir -p "$data_root"

slurm_logs=$data_root/_slurm_logs/
mkdir -p "$slurm_logs"

population=${POPULATION:-100}
generations=${GENERATIONS:-100}
learning=${LEARNING:-100}
population=${POPULATION:-100}
threads=${THREADS:-8}
duration=${SLURM_DURATION:-10:00:00}
partition=${SLURM_PARTITION:-batch}

tasks=${TASKS:-locomotion}
symmetries=${SYMMETRIES:-none}

echo "  Experiment: $exp"
echo "      Folder: $data_root"
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

jobs=.jobs.$(date +%s).slurm_array
rm .jobs.*.slurm_array

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

    [ -d $data_folder ] && continue
#    echo $job_name $data_folder >&2
    echo $data_folder \
      python -m aapets.g_cpg.main --seed $seed \
        --task $task --symmetry $symmetry \
        $args \
        --no-overwrite --threads $threads --data-folder $data_folder \
        --population-size $population --generations $generations --learning $learning
  done
done | nl -v0 -w1 -s ' ' > $jobs

njobs=$(wc -l < $jobs)
array=0-$((njobs-1))
log "Scheduling n=$njobs jobs (array=$array)"

sbatch -o "$slurm_logs/%x-%a.out" -e "$slurm_logs/%x-%a.err" <<EOF
#!/bin/bash

#SBATCH --job-name=$exp
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$threads
#SBATCH --mem=5G
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
