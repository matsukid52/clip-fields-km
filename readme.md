# CLIP-Fields-km

[[Paper]](https://arxiv.org/abs/2210.05663) [[Website]](https://mahis.life/clip-fields/) [[Code]](https://github.com/notmahi/clip-fields) [[Data]](https://osf.io/famgv) [[Video]](https://youtu.be/bKu7GvRiSQU)


https://user-images.githubusercontent.com/3000253/195213301-43eae6e8-4516-4b8d-98e7-633c607c6616.mp4

**Tl;dr** CLIP-Field is a novel weakly supervised approach for learning a semantic robot memory that can respond to natural language queries solely from raw RGB-D and odometry data with no extra human labelling. It combines the image and language understanding capabilites of novel vision-language models (VLMs) like CLIP, large language models like sentence BERT, and open-label object detection models like Detic, and with spatial understanding capabilites of neural radiance field (NeRF) style architectures to build a spatial database that holds semantic information in it.

## 環境構築
### 使用環境
ALIENWARE
GeForce GTX 2070 ,Turing　（CUDA SDK version：10.0-13.0,CUDA Version: 13.0まで対応） 
Driver Version:580.95.05　CUDA：12.4
PyTorch:2.4.1（2025現在のHSR）NumPy：1.22.2


1. リポジトリをクローン
```
git clone --recursive git@github.com:mkid52/clip-fields-km.git
cd clip-fields-km
```

2. Anacondaをインストール
```
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash /root/HSR/DL/Anaconda3-2025.06-0-Linux-x86_64.sh
source ~/anaconda3/bin/activate
#動作確認
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

5. インタラクティブチュートリアルと評価
依存パッケージのインストールが完了したら,[`demo/`](https://github.com/notmahi/clip-fields/tree/main/demo) ディレクトリ内にある.インタラクティブなチュートリアルおよび評価用ノートブックを実行することで,モデルの動作を確認したり,自分のデータで評価を行うことができる.
example.pyはgooglecolabのソースコードをまとめたプログラムである.


7. CLIP-Field の学習
依存環境をインストールした後,任意の [.r3d](https://record3d.app/) ファイルを使用して
train.py スクリプトを実行することで,サンプルデータを学習させることができる.
[sample data](https://osf.io/famgv) `nyu.r3d`を使用して試す場合は,以下を実行する.

```
python train.py dataset_path=nyu.r3d device=cuda
```

学習が完了すると,clip_implicit_model下に学習データが出力される.
```
cd clip_implicit_model
ls
#implicit_scene_label_model_latest.pt があればOK.
```

7. LSegを使用する場合
オープンラベルアノテーションの追加ソースとして, [LSeg demo model](https://github.com/isl-org/lang-seg#-try-demo-now)を利用する場合は,LSegのデモモデルをダウンロードし,以下のパスに配置する.

```
path_to_LSeg/checkpoints/demo_e200.ckpt
```
その後,以下のコマンドでLSegを有効にして学習を実行する.
```
python train.py dataset_path=nyu.r3d use_lseg=true
```

設定ファイルのカスタマイズ
利用可能な設定オプションは config/train.yaml に記載されている.特定のラベルセットで学習したい場合は,custom_labels フィールドを編集して使用するラベルを指定する.

* [CLIP](https://github.com/openai/CLIP) with [MIT License](https://github.com/openai/CLIP/blob/main/LICENSE)
* [Detic](https://github.com/facebookresearch/Detic/) with [Apache License 2.0](https://github.com/facebookresearch/Detic/blob/main/LICENSE)
* [Torch NGP](https://github.com/ashawkey/torch-ngp) with [MIT License](https://github.com/ashawkey/torch-ngp/blob/main/LICENSE)
* [LSeg](https://github.com/isl-org/lang-seg) with [MIT License](https://github.com/isl-org/lang-seg/blob/main/LICENSE)
* [Sentence BERT](https://www.sbert.net/) with [Apache License 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)
