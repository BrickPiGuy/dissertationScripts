#!/bin/bash

# === CONFIGURATION ===
DEVICE="/dev/nvme0n1p1"
MOUNT_POINT="/mnt/openllama_data"
DEVICE_LABEL="OPENLLAMA_DATA"
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo "🚀 Starting OpenLLAMA SSD setup..."

# === STEP 1: Create mount point ===
echo "📁 Creating mount point at $MOUNT_POINT..."
sudo mkdir -p $MOUNT_POINT

# === STEP 2: Optional format (uncomment to enable wipe) ===
read -p "⚠️ Do you want to format $DEVICE as ext4? This will erase ALL data on the SSD. [y/N]: " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    echo "🔄 Formatting $DEVICE as ext4..."
    sudo mkfs.ext4 -L $DEVICE_LABEL $DEVICE
fi

# === STEP 3: Get UUID and update fstab ===
UUID=$(sudo blkid -s UUID -o value $DEVICE)
echo "🔗 UUID for $DEVICE is $UUID"
FSTAB_ENTRY="UUID=$UUID $MOUNT_POINT ext4 defaults,nofail,uid=$USER_ID,gid=$GROUP_ID 0 2"

echo "📝 Adding to /etc/fstab for auto-mount..."
if ! grep -q "$UUID" /etc/fstab; then
    echo "$FSTAB_ENTRY" | sudo tee -a /etc/fstab
else
    echo "✅ Entry already exists in /etc/fstab"
fi

# === STEP 4: Mount the SSD ===
echo "📦 Mounting SSD..."
sudo mount -a

# === STEP 5: Create Hugging Face cache folders ===
echo "📁 Creating Hugging Face cache directories..."
mkdir -p $MOUNT_POINT/huggingface/transformers
mkdir -p $MOUNT_POINT/huggingface/datasets

# === STEP 6: Add Hugging Face environment to .bashrc ===
echo "🔧 Configuring Hugging Face environment..."
echo "export HF_HOME=$MOUNT_POINT/huggingface" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=\$HF_HOME/transformers" >> ~/.bashrc
echo "export HF_DATASETS_CACHE=\$HF_HOME/datasets" >> ~/.bashrc

# === STEP 7: Apply changes ===
echo "🔄 Reloading environment..."
source ~/.bashrc

echo "✅ SSD setup complete! Hugging Face cache will now use: $MOUNT_POINT/huggingface"
