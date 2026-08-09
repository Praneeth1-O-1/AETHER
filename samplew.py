from torchgeo.datasets import SEN12MS

dataset = SEN12MS(
    root="./datasets",
    split="train",
    download=True
)

sample = dataset[0]