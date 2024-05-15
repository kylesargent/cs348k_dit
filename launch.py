import submitit
import argparse
from train import main
from models import DiT_models

class Task:
    def __call__(self, args):
        submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
        main(args)


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()


    NUM_TASKS_PER_NODE = 4

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        timeout_min=60*36,
        # slurm_partition="viscam",
        # account="viscam",
        # slurm_additional_parameters={"nodelist": "viscam2,viscam5,viscam9"},
        nodes=1,
        gpus_per_node=NUM_TASKS_PER_NODE,
        tasks_per_node=NUM_TASKS_PER_NODE,
        mem_gb=64,
        cpus_per_task=8,
    )
    task = Task()
    job = executor.submit(task, args)
    print(job.job_id)