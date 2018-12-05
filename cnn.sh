rm sl*
sbatch --gres=gpu:1 --time=30 --wrap="python run.py"

while [ ! -e sl* ] # while slurm.out does not exist
do
    sleep 2
    squeue
done
sleep 2
cat sl*
