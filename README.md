
# 📚 Data
You could generate demonstrations by yourself using our provided expert policies.  Generated demonstrations are under `$YOUR_REPO_PATH/FlowPolicy/data/`.

# 🛠️ Usage
Scripts for generating demonstrations, training, and evaluation are all provided in the `scripts/` folder. 

The results are logged by `wandb`, so you need to `wandb login` first to see the results and videos.

1. Generate demonstrations by `gen_demonstration_adroit.sh` and `gen_demonstration_metaworld.sh`. See the scripts for details. For example:
    ```bash
    bash scripts/gen_demonstration_adroit.sh hammer
    ```
    This will generate demonstrations for the `hammer` task in Adroit environment. The data will be saved in `FlowPolicy/data/` folder automatically.

2. Train and evaluate a policy with behavior cloning. For example:
    ```bash
    bash scripts/train_policy.sh flowpolicy adroit_hammer Short 0 0
    ```
    This will train a flowpolicy policy on the `hammer` task in Adroit environment using point cloud modality.

3. Evaluate a saved policy or use it for inference. Please set  For example:
    ```bash
    bash scripts/eval_policy.sh flowpolicy adroit_hammer Short 0 0
    ```
     **Note: the evaluation script is only provided for deployment/inference. For benchmarking, please use the results logged in wandb during training.**
