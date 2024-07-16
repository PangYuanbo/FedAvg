

# Project Name

This project uses PyTorch as the primary deep learning framework. Below are the instructions for installing PyTorch on different operating systems.

## Installation

### macOS

On macOS, you can install PyTorch using Conda or Pip with the following commands:

#### Using Conda

```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

#### Using Pip

```bash
pip3 install torch torchvision torchaudio
```

### Windows

On Windows, you can install PyTorch using Pip or Conda with the following commands:

#### Using Pip

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Using Conda

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Other Dependencies

Make sure to install all other required dependencies for the project. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Running the Project

After installing all dependencies, you can run the main script of the project:

```bash
python main.py
```

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).


By including these instructions in the README file, you provide clear guidance on how to install PyTorch and other dependencies on both macOS and Windows, making it easier for users to set up the project.

# FedAvg
## 2NN (CPU:M2)
｜C | IsIID | B  | Rounds | E |L|Time｜Accuracy|
|--|-------|----|--------|---|---|---|
|0| True  | ∞  | 1455   | 1 |0.1||34.02｜92.45%|
｜0.1| True  | ∞  | 1474   | 1 |0.1||318.42｜96.26%|
｜0.2| True  | ∞  | 1658   | 1 |0.1||683.4｜96.7%|
|0| True  | 10 | 316    | 1 |0.1||---｜
｜0.1| True  | 10 | 87     | 1 |0.1||340｜98.23%|
｜0.2| True  | 10 | 77     | 1 |0.1||---｜
｜0.5| True  | 10 | 75     | 1 |0.1||---｜
｜1| True  | 10 | 70     | 1 |0.1||---｜

｜C | IsIID | B  | Rounds | E |L|Time｜Accuracy|
|--|-------|----|--------|---|---|---|
|0| False | ∞  | 4278   | 1 |0.1||34.02｜92.45%|
｜0.1| False  | ∞  | 1796   | 1 |0.1||318.42｜96.26%|
｜0.2| False  | ∞  | 1528   | 1 |0.1||683.4｜96.7%|
|0| False  | 10 | 3275   | 1 |0.1||---｜
｜0.1| False  | 10 | 664    | 1 |0.1||---｜
｜0.2| False  | 10 | 619    | 1 |0.1||---｜
｜0.5| False  | 10 | 443    | 1 |0.1||---｜
｜1| False  | 10 | 380    | 1 |0.1||---｜


## CNN (CUDA:4070tis)
｜C | IsIID | B | Rounds | E |L|Time
|--|-------|---|--------|---|---|---|
|0| True  | ∞ | 387    | 5 |0.1||131.44｜97.42%|
｜0.1| True  | ∞ | 339    | 5 |0.1||---｜
｜0.2| True  | ∞ | 337    | 5 |0.1||---｜
｜0.5| True  | ∞ | 164    | 5 |0.1||---｜
｜1| True  | ∞ | 246    | 5 |0.1||---｜
|0| True  | ∞ | 50     | 5 |0.1||131.44｜97.42%|
｜0.1| True  | ∞ | 18     | 5 |0.1||---｜
｜0.2| True  | ∞ | 18     | 5 |0.1||---｜
｜0.5| True  | ∞ | 18     | 5 |0.1||---｜
｜1| True  | ∞ | 16     | 5 |0.1||---｜

｜C | IsIID | B | Rounds | E |L|Time
|--|-------|---|--------|---|---|---|
|0| False | ∞ | 1181   | 5 |0.1||131.44｜97.42%|
｜0.1| False  | ∞ | 1100   | 5 |0.1||---｜
｜0.2| False  | ∞ | 978    | 5 |0.1||---｜
｜0.5| False  | ∞ | 1067   | 5 |0.1||---｜
｜1| False  | ∞ | ---    | 5 |0.1||---｜
|0| False | ∞ | 956    | 5 |0.1||131.44｜97.42%|
｜0.1| False  | ∞ | 206    | 5 |0.1||---｜
｜0.2| False  | ∞ | 200    | 5 |0.1||---｜
｜0.5| False  | ∞ | 261    | 5 |0.1||---｜
｜1| False  | ∞ | 97     | 5 |0.1||---｜
