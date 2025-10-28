# CLIP-Fields-km

[[Paper]](https://arxiv.org/abs/2210.05663) [[Website]](https://mahis.life/clip-fields/) [[Code]](https://github.com/notmahi/clip-fields) [[Data]](https://osf.io/famgv) [[Video]](https://youtu.be/bKu7GvRiSQU)


https://user-images.githubusercontent.com/3000253/195213301-43eae6e8-4516-4b8d-98e7-633c607c6616.mp4

**Tl;dr** CLIP-Field is a novel weakly supervised approach for learning a semantic robot memory that can respond to natural language queries solely from raw RGB-D and odometry data with no extra human labelling. It combines the image and language understanding capabilites of novel vision-language models (VLMs) like CLIP, large language models like sentence BERT, and open-label object detection models like Detic, and with spatial understanding capabilites of neural radiance field (NeRF) style architectures to build a spatial database that holds semantic information in it.

## 環境構築

1.リポジトリをクローン
```
git clone --recursive git@github.com:mkid52/clip-fields-km.git
cd clip-fields-km
```

2. Anacondaをインストール
```
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash /root/HSR/DL/Anaconda3-2025.06-0-Linux-x86_64.sh
source ~/anaconda3/bin/activate
#インストールされているか確認
conda --version
```

・Anacondaのコマンド例
```
conda info -e
conda remove -n cf --all
```

3. 仮想環境作成と依存環境をインストール
```
conda create -n cf python=3.8
conda activate cf
conda install -y pytorch==2.4.1 torchvision torchaudio -c pytorch -c nvidia
pip install -r requirements.txt
```

4. CUDA設定とgridencoderビルド
gridencoder はC++/CUDA拡張を含むため、環境に合わせて再ビルドが必要です。
他環境でビルド済みの .so を使用すると、以下のようなエラーが発生します：
```vbnet
ImportError: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
```
これを防ぐため、自分の環境で再ビルドする

```
cd gridencoder
# CUDAパスを設定（例: CUDA 11.8）
which nvcc
# 例: /usr/local/cuda-11.8/bin/nvcc が返る場合
export CUDA_HOME=/usr/local/cuda-11.8

# ninjaをインストール（PyTorch拡張ビルドに必須）
conda install -y ninja

# GridEncoderを再ビルド＆インストール
pip install .
#動作確認コマンド
python -c "import gridencoder"
cd ..
```

上記を実行すると、gridencoder が現在の環境でコンパイル・登録され、
python -c "import gridencoder" が正常に動作する

```
```

5. インタラクティブチュートリアルと評価
We have an interactive tutorial and evaluation notebook that you can use to explore the model and evaluate it on your own data. You can find them in the [`demo/`](https://github.com/notmahi/clip-fields/tree/main/demo) directory, that you can run after installing the dependencies.

6. Training a CLIP-Field directly
Once you have the dependencies installed, you can run the training script `train.py` with any [.r3d](https://record3d.app/) files that you have! If you just want to try out a sample, download the [sample data](https://osf.io/famgv) `nyu.r3d` and run the following command.

```
python train.py dataset_path=nyu.r3d
```

If you want to use LSeg as an additional source of open-label annotations, you should download the [LSeg demo model](https://github.com/isl-org/lang-seg#-try-demo-now) and place it in the `path_to_LSeg/checkpoints/demo_e200.ckpt`. Then, you can run the following command.

```
python train.py dataset_path=nyu.r3d use_lseg=true
```

You can check out the `config/train.yaml` for a list of possible configuration options. In particular, if you want to train with any particular set of labels, you can specify them in the `custom_labels` field in `config/train.yaml`.


## Acknowledgements
We would like to thank the following projects for making their code and models available, which we relied upon heavily in this work.
* [CLIP](https://github.com/openai/CLIP) with [MIT License](https://github.com/openai/CLIP/blob/main/LICENSE)
* [Detic](https://github.com/facebookresearch/Detic/) with [Apache License 2.0](https://github.com/facebookresearch/Detic/blob/main/LICENSE)
* [Torch NGP](https://github.com/ashawkey/torch-ngp) with [MIT License](https://github.com/ashawkey/torch-ngp/blob/main/LICENSE)
* [LSeg](https://github.com/isl-org/lang-seg) with [MIT License](https://github.com/isl-org/lang-seg/blob/main/LICENSE)
* [Sentence BERT](https://www.sbert.net/) with [Apache License 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)
