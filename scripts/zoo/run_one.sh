#!/bin/bash

usage(){
  echo "Usage: $0 <name> <seeds> [...ARGS]"
  echo "          name is the name of the experiment (top-level folder)"
  echo "          seeds will populate SLURM's array field"
  echo "          any other argument are passed through to the executable"
}

if [ $# -lt 3 ]
then
  usage
  exit 1
fi

name=$1
seeds=$2
shift 2

venv=ariel-venv

data_root=$HOME/data/zoo/
mkdir -p "$data_root"

slurm_logs=$data_root/slurm_logs/$name/
mkdir -p "$slurm_logs"

job_name=cma-$name

slurm_logs_base="$slurm_logs/run-%a"

duration=${SLURM_DURATION:-1:00:00}
threads=${THREADS:-8}
partition=${SLURM_PARTITION:-batch}
budget=${BUDGET:-10000}

printf "Running cma/$name[$seeds] on partition $partition with $threads threads for $duration -> "

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
data_folder=$data_root$name/run-\$seed

if [ -n "$SILENT_SKIP_EXISTING" ] && [ -d "\$data_folder" ]
then
  echo "Folder \$data_folder already exists. Silently aborting"
  exit 0
fi

. ${PYENV_ROOT}/versions/$venv/bin/activate

date
echo "Seed is \$seed"
echo "Saving data to \$data_folder"
echo "Additional arguments: $@"

#module load graphviz/12.2.1

export MUJOCO_GL=egl
python -m aapets.zoo.evolve --no-overwrite --no-symlink-last --budget $budget --duration 15 \
 --threads $threads --seed \$seed --data-folder \$data_folder $@

for ext in out err
do
  mv -v $slurm_logs/run-\$seed.\$ext \$data_folder/slurm.\$ext
done

rmdir -p --ignore-fail-on-non-empty $slurm_logs

EOF
