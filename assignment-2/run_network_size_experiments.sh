#!/usr/bin/env bash

# Define model sizes
HIDDEN_SIZES=(64 128 256 512)
LAYERS=(2 2 3 4)
HEADS=(2 4 4 8)
SIZES=(xs s m xl)

LR=1e-4
BATCH_SIZE=64
LOG_FILE="dimension_log.csv"

echo "dimension,hidden_size,layers,heads,recall@10,ndcg@10" > $LOG_FILE

for i in ${!SIZES[@]}
do
  size=${SIZES[$i]}
  hidden_size=${HIDDEN_SIZES[$i]}
  layers=${LAYERS[$i]}
  heads=${HEADS[$i]}
  MODEL_FILE="bert4rec_${size}.pth"

  echo "▶️ Training model size: $size (hidden_size=$hidden_size, layers=$layers, heads=$heads)"

  python3 train.py \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --hidden_size $hidden_size \
    --num_layers $layers \
    --num_heads $heads \
    --model_save_path $MODEL_FILE

  echo "✅ Training complete. Now evaluating..."

  OUTPUT=$(python3 evaluate.py \
    --hidden_size $hidden_size \
    --num_layers $layers \
    --num_heads $heads \
    --model_path $MODEL_FILE 2>&1)

  RECALL=$(echo "$OUTPUT" | grep "Recall@10" | awk '{print $2}')
  NDCG=$(echo "$OUTPUT" | grep "NDCG@10" | awk '{print $2}')

  echo "$size,$hidden_size,$layers,$heads,$RECALL,$NDCG" >> $LOG_FILE
  echo "📝 Logged $size run to $LOG_FILE"
done

echo "🎉 All network size experiments completed!"
