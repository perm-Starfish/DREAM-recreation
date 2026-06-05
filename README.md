目前主要執行檔是 run_minimal_krono.py會執行以下流程：

1.  讀取 KronoDroid malware dataset
2.  選出樣本數最多的 malware families
3.  使用 hold-out family 模擬 inter-class concept drift
4.  訓練 malware family classifier
5.  訓練 DREAM-style contrastive autoencoder drift detector
6.  計算 drift score
7.  選出 top-k drift samples
8.  執行 concept-space explanation
9.  進行 joint adaptation
10. 評估 accuracy 與 macro-F1

# 環境設定

建立 conda 環境：

conda create -n dream-repro python=3.10 -y

conda activate dream-repro

安裝需要的套件：

pip install torch torchvision torchaudio

pip install numpy pandas scikit-learn tqdm matplotlib

如果要使用 NVIDIA GPU，可以安裝 CUDA 版本的 PyTorch：

pip uninstall torch torchvision torchaudio -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 專案資料夾結構

DREAM_reproduction/

├── data/

│ └── emu_malware_v1.zip

├── checkpoints/

├── results/

├── src/

└── run_minimal_krono.py

# 使用的資料集

原論文使用的資料集是Drebin和MalRadar，但因為沒有下載權限，這裡改用KronoDroid。原始實驗使用的惡意程式家族包含：

FakeInstaller

DroidKungFu

Plankton

GingerMaster

BaseBridge

Iconosys

Kmin

FakeDoc


目前用KronoDroid資料集實驗選出的前8個惡意程式家族是：

Airpush/StopSMS: 6521 samples

Boxer: 3557 samples

Malap: 2574 samples

FakeInst: 2158 samples

Agent: 1837 samples

Locker/SLocker Ransomware: 1822 samples

BankBot: 1241 samples

FakeApp: 1064 samples

# 重要參數

classifier_epochs: classifier訓練epoch數

detector_epochs: DREAM detector訓練epoch數

adapt_epochs: adaptation訓練epoch數

batch_size: batch size

learning_rate: learning rate

lambda_rec: reconstruction loss權重

lambda_sep: contrastive separation loss權重

lambda_rel: concept reliability loss權重

lambda_pre: concept presence losS權重

sep_margin: contrastive loss margin

ae_hidden_dim: autoencoder hidden dimension

ae_encoding_dim: autoencoder latent dimension

eta: adaptation 時 detector learning rate的縮小比例

num_families: 使用的 malware family數量

min_family_samples: family最少樣本數門檻
