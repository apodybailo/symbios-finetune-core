#!/bin/bash

# ПЕРЕД ЗАПУСКОМ: переконайся, що у тебе є HuggingFace token
# і встановлено все з requirements.txt

# Назва твоєї моделі на Hugging Face
MODEL_NAME="APAndreyAI/symbios-mistral-lora"
DATA_PATH="./data/training_data.jsonl"
OUTPUT_DIR="./outputs"

# LoRA параметри (можна редагувати під себе)
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
RANK=16
ALPHA=32
EPOCHS=3
LR=5e-5

echo "🔁 Запускаємо Fine-tune LoRA..."
python train_lora.py \
  --base_model $BASE_MODEL \
  --dataset_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --lora_rank $RANK \
  --lora_alpha $ALPHA \
  --epochs $EPOCHS \
  --learning_rate $LR

if [ $? -ne 0 ]; then
  echo "❌ Помилка при тренуванні."
  exit 1
fi

echo "✅ Тренування завершено. Пушимо на HuggingFace..."

huggingface-cli login --token $HF_TOKEN

transformers-cli repo create $MODEL_NAME --type model || true

cd $OUTPUT_DIR
git init
git remote add origin https://huggingface.co/$MODEL_NAME
git checkout -b main
git add .
git commit -m "Initial LoRA push"
git push origin main --force

echo "🚀 Модель $MODEL_NAME успішно запушена на Hugging Face."
