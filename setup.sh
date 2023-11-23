mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

export PATH=~/miniconda3/bin:~/miniconda3/scripts:$PATH

conda init bash
conda init zsh
conda install python=3.11.5

conda create -n tts
conda activate tts
conda install cudatoolkit=11.0 -y

pip install torch sentencepiece datasets bitsandbytes peft accelerate scipy pipx
pip install huggingface_hub=0.2.1
pip install huggingface_hub[torch]