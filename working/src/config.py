import os


class Config:
    """
        Here we save the configuration for the experiments
    """
    # dirs
    base_dir = os.path.abspath('../')
    data_dir = os.path.join(base_dir, 'input')
    working_dir = os.path.join(base_dir, 'working')
    submissions_dir = os.path.join(base_dir, 'submissions')
    models_dir = os.path.join(base_dir, 'models')
    logs_dir = os.path.join(base_dir, 'logs')
    # Hparams
    train_batch_size = 32
    test_batch_size = 32
    base_model = "resnet18"
    learning_rate = 1e-3
    device = "cuda"  # cuda -> GPU, "cpu"->CPU, "tpu"->TPU


if __name__ == '__main__':
    print(Config.__dict__)
