Запуск:

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision thop pandas tqdm

python baseline.py --datasets CIFAR10 SVHN EMNIST --models vgg16 resnet18 --epochs 150 --batch-size 200