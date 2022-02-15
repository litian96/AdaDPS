# Private Adaptive Optimization with Side Information


This repository contains the code and experiments for the manuscript:

> [Private Adaptive Optimization with Side Information](https://arxiv.org/abs/2202.05963)

Adaptive optimization methods have become the default solvers for many machine learning tasks.
Unfortunately, the benefits of adaptivity may degrade when training with differential privacy, as the noise
added to ensure privacy reduces the effectiveness of the adaptive preconditioner. To this end, we propose
AdaDPS, a general framework that uses non-sensitive side information to precondition the gradients,
allowing the effective use of adaptive methods in private settings. We formally show AdaDPS reduces the
amount of noise needed to achieve similar privacy guarantees, thereby improving optimization performance.
Empirically, we leverage simple and readily available side information to explore the performance of AdaDPS
in practice, comparing to strong baselines in both centralized and federated settings. Our results show
that AdaDPS improves accuracy by 7.7% (absolute) on average—yielding state-of-the-art privacy-utility
trade-offs on large-scale text and image benchmarks.




### Prepare Datasets

* MNIST with autoencoder: under the folder `AE/`, the code will automatically download MNIST 


* IMDB
	* Download the data in this [Google Drive link](https://drive.google.com/file/d/1yt5JW_Pi4Dy8TR9E1B9w2HFpIBZYNuz8/view?usp=sharing), unzip it
	* logistic regression with BoW features: put `imdb_10000d_train_bow.npz` and `imdb_10000d_test_bow.npz` under `imdb/data/`
	* LSTM with raw feature: put `imdb_10000d_test.npz` and `imdb_10000d_train.npz` under `imdb/data`

	
* StackOverflow (federated dataset)

```
cd federated/data/stackoverflow
mkdir data/test/test_np
mkdir data/train/train_np
python create_data.py
python preprocess.py 
```

* StackOverflow (centralized dataset)

```
cd stackoverflow_centralized/
python generate_data.py
```

### Usage

Each dataset has a separate folder. Under each folder, the algorithms are implemented under `trainers` for centralized training, or `flearn/trainers` for the federated learning application. File names indicate method names. AdaDPS is implemented in SGD or DP-SGD with the options `use_public` or `scale` on, specified in `run.sh`. We provide some concrete examples of running the linear regression toy problem or IMDB below.



### Running Examples

* linear regression toy problem (Figure 3 in the paper)

```
cd toy
python3 main.py --method sgd --scale 0 --lr 0.03  --iters 40000 --eval_every_iter 1000 # SGD
python3 main.py --method sgd --scale 1 --lr 0.003 --iters 40000 --eval_every_iter 1000 # AdaS
python3 main.py --method dp-sgd --scale 0 --clipping_bound 0.5 --lr 0.1 --sigma 0.7 --iters 2250 # DP-SGD
python3 main.py --method dp-sgd --scale 1 --clipping_bound 0.5 --lr 0.2 --sigma 0.7 --iters 2250 # AdaDPS
```

* IMDB (BoW)
	* Prepare data as described previously, and put the four files under `imdb/data`
	* Run

	
	```
	bash run1.sh # DP-SGD 
	bash run2.sh # AdaDPS w/ public data
	```
	
See hyperparameter tuning and their values reported in the manuscript (Appendix B).