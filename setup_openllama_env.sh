#!/bin/bash

echo "🔧 Updating and upgrading system packages..."
sudo apt update && sudo apt upgrade -y

echo "📦 Installing build dependencies and Python environment tools..."
sudo apt install -y python3-pip python3-venv git build-essential

echo "🐍 Creating Python virtual environment..."
python3 -m venv ~/openllama_env
source ~/openllama_env/bin/activate

echo "⬆️ Upgrading pip and installing Python packages..."
pip install --upgrade pip

echo "🔥 Installing JetPack-compatible PyTorch + torchvision..."
pip install torch==2.0.0+nv23.05 torchvision==0.15.1+nv23.05 -f https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/

echo "🧠 Installing Hugging Face libraries..."
pip install transformers datasets tokenizers accelerate

echo "📊 Installing model optimization & evaluation libraries..."
pip install onnx onnxruntime

echo "📈 Installing data science stack..."
pip install numpy pandas matplotlib scikit-learn

echo "⚙️ Adding TensorRT tools to PATH..."
export PATH=$PATH:/usr/src/tensorrt/bin

echo "🔍 Installing NVIDIA Nsight tools and monitoring..."
sudo apt install -y nvidia-nsight nvidia-nsight-systems
pip install nvitop

echo "📂 Cloning OpenLLaMA repository..."
git clone https://github.com/openlm-research/open_llama.git ~/open_llama
cd ~/open_llama
pip install -r requirements.txt

echo "✅ Environment setup complete."
echo "💡 To activate this environment later, run:"
echo "source ~/openllama_env/bin/activate"

