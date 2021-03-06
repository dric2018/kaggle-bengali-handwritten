import os


class Config:
    """
        Here we save the configuration for the experiments
    """
    # dirs
    base_dir = os.path.abspath('/home/zeusdric/Dric/kaggle/bengali-ai')
    data_dir = os.path.join(base_dir, 'input')
    images_dir = os.path.join(data_dir, 'images')
    working_dir = os.path.join(base_dir, 'working')
    submissions_dir = os.path.join(base_dir, 'submissions')
    models_dir = os.path.join(base_dir, 'models')
    logs_dir = os.path.join(base_dir, 'logs')

    # Hparams
    seed_val = 2021
    height = 137
    width = 236
    resize_shape = (100, 100)
    train_batch_size = 512
    test_batch_size = 512
    epochs = 2
    # seresnet50, seresnet152, efficientnet_b3, resnet101, resnext50_32x4d
    base_model = "resnet34"
    learning_rate = 0.02
    num_workers = 4
    device = "cuda"  # cuda -> GPU, "cpu"->CPU, "tpu"->TPU
    data_transform = "basic"  # fmix, cutmix, mixup


if __name__ == '__main__':
    print(Config.__dict__)
