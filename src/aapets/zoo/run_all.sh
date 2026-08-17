#!/bin/bash

usage(){
    echo "Usage: $0 <seeds> [ARGS...]"
    echo "       Schedules lots of runs under folder zoo with <seeds>."
    echo "       Seeds are expanded (via python) and additional arguments are forwarded to"
    echo "        the evolution"
}

# Fail as quick as possible
set -euo pipefail

if [ $# -lt 1 ]
then
  echo "Not enough arguments"
  usage
fi

name=zoo
seeds=$1
shift 1

log(){
  echo "[$(date)] $@"
}

log "Preparing jobs list for seeds=$seeds"

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

data_root=$HOME/data/$name
mkdir -p "$data_root"

slurm_logs=$data_root/_slurm_logs/
mkdir -p "$slurm_logs"

budget=${BUDGET:-10000}
threads=${THREADS:-8}
duration=${SLURM_DURATION:-10:00:00}
partition=${SLURM_PARTITION:-batch}
limits=${LIMITS:-}

bodies=${BODIES:-all}
if [[ $bodies == "all" ]]
then
  bodies=$(python -m aapets.zoo.evolve --print-canonical-bodies)
fi

if [[ -n $limits ]]
then
  limits="%$limits"
fi

echo " Experiment: $name"
echo "     Folder: $data_root"
echo "    Threads: $threads"
echo "     Limits: $limits"
echo "   Duration: $duration"
echo "  Partition: $partition"
echo "     Budget: $budget"
echo "     Bodies: $bodies"

read -p "All good? [Yy]es " -n 1 -r go
[[ "$go" =~ ^[Yy]$ ]] || (echo; exit 2)
echo

jobs=.jobs.$name.$(date +%s).slurm_array
# rm -f .jobs.$name.*.slurm_array

(
  for body in $bodies
  do
    echo $body
  done
) | while read body
do
  for seed in $expanded_seeds
  do
    job_name=$body/run-$seed
    data_folder=$data_root/$job_name

    [ -d $data_folder ] && continue
#    echo $job_name $data_folder >&2
    echo $data_folder \
      python -m aapets.zoo.evolve --seed $seed --body $body \
        --no-overwrite --no-symlink-last \
        --threads $threads --data-folder $data_folder \
        --budget $budget --duration 15 \
        $@
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

jobname=cma-zoo

sbatch -o "$slurm_logs/$jobname-%a.out" -e "$slurm_logs/$jobname-%a.err" <<EOF
#!/bin/bash

#SBATCH --job-name=$jobname
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$threads
#SBATCH --mem=5G
#SBATCH --array=$array$limits
#SBATCH --time=$duration
#SBATCH --exclude=node01

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
  mv -v $slurm_logs/$jobname-\$task_id.\$ext \$folder/slurm.\$ext
done

rmdir -p --ignore-fail-on-non-empty $slurm_logs

EOF
