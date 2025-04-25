#!/usr/bin/env bash

# Based on previous results: best performing model is `xl` (512 hidden, 4 layers, 8 heads)

SEQ_LENS=(50 75 100 150)
HIDDEN_SIZE=512
LAYERS=4
HEADS=8
LR=1e-4
BATCH_SIZE=64
SCHEDULER="cosine"  # Options: cosine, step, plateau
LOG_FILE="seq_len_ablation_log.csv"

echo "seq_len,hidden_size,layers,heads,scheduler,recall@10,ndcg@10" > $LOG_FILE

for seq_len in "${SEQ_LENS[@]}"
do
  MODEL_FILE="bert4rec_seq${seq_len}.pth"

  echo "▶️ Training seq_len=$seq_len (hidden_size=$HIDDEN_SIZE, layers=$LAYERS, heads=$HEADS, scheduler=$SCHEDULER)"

  python3 train.py \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $LAYERS \
    --num_heads $HEADS \
    --seq_len $seq_len \
    --scheduler $SCHEDULER \
    --model_save_path $MODEL_FILE

  echo "✅ Training complete. Now evaluating..."

  OUTPUT=$(python3 evaluate.py \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $LAYERS \
    --num_heads $HEADS \
    --seq_len $seq_len \
    --model_path $MODEL_FILE 2>&1)

  RECALL=$(echo "$OUTPUT" | grep "Recall@10" | awk '{print $2}')
  NDCG=$(echo "$OUTPUT" | grep "NDCG@10" | awk '{print $2}')

  echo "$seq_len,$HIDDEN_SIZE,$LAYERS,$HEADS,$SCHEDULER,$RECALL,$NDCG" >> $LOG_FILE
  echo "📝 Logged seq_len=$seq_len run to $LOG_FILE"
done

echo "🎉 All sequence length experiments completed!"
