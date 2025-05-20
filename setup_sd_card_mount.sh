#!/bin/bash

# === Configuration ===
MOUNT_POINT="/mnt/openllama_sd"
DEVICE_LABEL="OPENLLAMA_SD"

# === Step 1: Create mount point ===
sudo mkdir -p $MOUNT_POINT

# === Step 2: Identify SD card device (manual step for safety) ===
echo "📝 Please insert your SD card and run 'lsblk' to identify its device name (e.g., /dev/mmcblk1p1)."
read -p "Enter the device name (e.g., /dev/mmcblk1p1): " DEVICE

# === Step 3: Format as exFAT (non-destructive if already formatted correctly) ===
echo "🔄 Formatting $DEVICE as exFAT..."
sudo mkfs.exfat -n $DEVICE_LABEL $DEVICE

# === Step 4: Add to /etc/fstab for auto-mounting ===
UUID=$(sudo blkid -s UUID -o value $DEVICE)
echo "📌 UUID of device is $UUID"
echo "Adding to /etc/fstab..."
echo "UUID=$UUID $MOUNT_POINT exfat defaults,nofail,uid=1000,gid=1000,dmask=027,fmask=137 0 0" | sudo tee -a /etc/fstab

# === Step 5: Mount the SD card ===
echo "🚀 Mounting $DEVICE at $MOUNT_POINT"
sudo mount -a

# === Step 6: Set Hugging Face cache environment (append to .bashrc) ===
echo "✅ Adding Hugging Face cache paths to ~/.bashrc"
echo "export HF_HOME=$MOUNT_POINT/huggingface" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=$HF_HOME/transformers" >> ~/.bashrc
echo "export HF_DATASETS_CACHE=$HF_HOME/datasets" >> ~/.bashrc

echo "🎉 Done! SD card will auto-mount and use this directory for Hugging Face model + dataset caching."
