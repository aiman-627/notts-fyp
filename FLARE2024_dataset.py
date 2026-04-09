from huggingface_hub import snapshot_download

local_dir = "./data"
snapshot_download(
    repo_id="FLARE-MedFM/FLARE-Task2-LaptopSeg",
    repo_type="dataset",
    allow_patterns=["coreset_train_50_random/*","train_gt_label/*","validation/*"],
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
