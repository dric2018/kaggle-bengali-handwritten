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
    resize_shape = (150, 150)
    train_batch_size = 1024
    test_batch_size = 1024
    epochs = 50
    # seresnet50, seresnet152, efficientnet_b3, resnet101, resnext50_32x4d
    base_model = "resnet34"
    learning_rate = 0.4
    num_workers = 4
    device = "cuda"  # cuda -> GPU, "cpu"->CPU, "tpu"->TPU


if __name__ == '__main__':
    print(Config.__dict__)
