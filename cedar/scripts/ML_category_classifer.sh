#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:v100l:2   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --time=8:00:00
#SBATCH --account=def-hefeeda
#SBATCH --mail-user=omossad@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/omossad/projects/def-hefeeda/omossad/cmpt980/cedar/output/test/job.out
#SBATCH -e /home/omossad/projects/def-hefeeda/omossad/cmpt980/cedar/output/test/job.err

module load cuda
module load cudnn
module load python
cd /home/omossad/projects/def-hefeeda/omossad/cmpt980/
source venv/bin/activate
cd /home/omossad/projects/def-hefeeda/omossad/cmpt980/code/
python ML_category_classifier.py train features.txt test
