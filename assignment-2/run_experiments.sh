#!/bin/bash

LEARNING_RATES=(1e-4 3e-4 5e-4 7e-4)
BATCH_SIZES=(64 128 256)

LOG_FILE="experiment_log.csv"

echo "run,learning_rate,batch_size,ndcg@10" > $LOG_FILE
run_id=1

for lr in "${LEARNING_RATES[@]}"
do
  for bs in "${BATCH_SIZES[@]}"
  do
    echo "▶️ Running experiment $run_id: LR=$lr, BS=$bs"
    
    # Train the model and capture logs
    OUTPUT=$(python3 train.py --lr $lr --batch_size $bs 2>&1)

    # Extract final validation NDCG@10 from the log
    FINAL_NDCG=$(echo "$OUTPUT" | grep "Validation NDCG@10" | tail -1 | awk '{print $3}')
    
    echo "$run_id,$lr,$bs,$FINAL_NDCG" >> $LOG_FILE
    echo "✅ Logged run $run_id"
    ((run_id++))
  done
done

echo "🎉 All experiments completed. Results saved to $LOG_FILE"
