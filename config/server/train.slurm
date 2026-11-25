#!/bin/bash
#SBATCH --job-name=mfvlr
#SBATCH --output=logs/mfvlr_%j.out
#SBATCH --error=logs/mfvlr_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -e
cd "$SLURM_SUBMIT_DIR"
source "/datastore/$USER/miniconda3/etc/profile.d/conda.sh"
conda activate yolo363
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -u -c "from ultralytics import YOLO; YOLO('weights/yolov8n.pt').train(cfg='config/server/vehicle_nano_train.yaml')"
python -u -c "from ultralytics import YOLO; YOLO('weights/yolov8s.pt').train(cfg='config/server/vehicle_small_train.yaml')"
python -u -c "from ultralytics import YOLO; YOLO('weights/yolov8n.pt').train(cfg='config/server/traffic_light_nano_train.yaml')"
python -u -c "from ultralytics import YOLO; YOLO('weights/yolov8s.pt').train(cfg='config/server/traffic_light_small_train.yaml')"
