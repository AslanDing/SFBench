# example for commands

for name in S_0 S_1 S_2 S_3 S_4 S_5 S_6 S_7
do
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method mlp --lr 0.0001 --weight_decay 0.000001 --epoches 15 --batchsize 64 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method mlp --lr 0.0001 --weight_decay 0.000001 --epoches 15 --batchsize 64 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method mlp --lr 0.0001 --weight_decay 0.000001 --epoches 15 --batchsize 64 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method mlp --lr 0.0001 --weight_decay 0.000001 --epoches 15 --batchsize 64 --seed 2025

  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method lstm --lr 0.0001 --weight_decay 0.00005 --epoches 50 --batchsize 64 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method lstm --lr 0.0001 --weight_decay 0.00005 --epoches 50 --batchsize 64 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method lstm --lr 0.0001 --weight_decay 0.00005 --epoches 50 --batchsize 64 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method lstm --lr 0.0001 --weight_decay 0.00005 --epoches 50 --batchsize 64 --seed 2025

  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method tcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method tcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method tcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method tcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025

  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method gcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method gcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method gcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method gcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
done




for name in S_0 S_1 S_2 S_3 S_4 S_5 S_6 S_7
do
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method mlp --lr 0.0001 --weight_decay 0.000001 --epoches 15 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method mlp --lr 0.0001 --weight_decay 0.000001 --epoches 15 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method mlp --lr 0.0001 --weight_decay 0.000001 --epoches 15 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method mlp --lr 0.0001 --weight_decay 0.000001 --epoches 15 --batchsize 64 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method lstm --lr 0.0001 --weight_decay 0.00005 --epoches 50 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method lstm --lr 0.0001 --weight_decay 0.00005 --epoches 50 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method lstm --lr 0.0001 --weight_decay 0.00005 --epoches 50 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method lstm --lr 0.0001 --weight_decay 0.00005 --epoches 50 --batchsize 64 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method tcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method tcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method tcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method tcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method gcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method gcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method gcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method gcn --lr 0.001 --weight_decay 0.0000001 --epoches 50 --batchsize 256 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method AutoTimes --lr 0.0005 --weight_decay 0.0 --epoches 10 --batchsize 128 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method AutoTimes --lr 0.0005 --weight_decay 0.0 --epoches 10 --batchsize 128 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method AutoTimes --lr 0.0005 --weight_decay 0.0 --epoches 10 --batchsize 128 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method AutoTimes --lr 0.0005 --weight_decay 0.0 --epoches 10 --batchsize 128 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method DeepAR --lr 0.001 --weight_decay 0.0 --epoches 20 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method DeepAR --lr 0.001 --weight_decay 0.0 --epoches 20 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method DeepAR --lr 0.001 --weight_decay 0.0 --epoches 20 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method DeepAR --lr 0.001 --weight_decay 0.0 --epoches 20 --batchsize 64 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method DilatedRNN --lr 0.001 --weight_decay 0.0 --epoches 40 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method DilatedRNN --lr 0.001 --weight_decay 0.0 --epoches 40 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method DilatedRNN --lr 0.001 --weight_decay 0.0 --epoches 40 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method DilatedRNN --lr 0.001 --weight_decay 0.0 --epoches 40 --batchsize 64 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method FourierGNN --lr 0.00001 --weight_decay 0.0 --epoches 100 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method FourierGNN --lr 0.00001 --weight_decay 0.0 --epoches 100 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method FourierGNN --lr 0.00001 --weight_decay 0.0 --epoches 100 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method FourierGNN --lr 0.00001 --weight_decay 0.0 --epoches 100 --batchsize 32 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method itransformer --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method itransformer --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method itransformer --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method itransformer --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 32 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method NLinear --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method NLinear --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method NLinear --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method NLinear --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method ModernTCN --lr 0.0001 --weight_decay 0.0 --epoches 100 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method ModernTCN --lr 0.0001 --weight_decay 0.0 --epoches 100 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method ModernTCN --lr 0.0001 --weight_decay 0.0 --epoches 100 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method ModernTCN --lr 0.0001 --weight_decay 0.0 --epoches 100 --batchsize 256 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method PatchTST --lr 0.0001 --weight_decay 0.0 --epoches 100 --batchsize 128 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method PatchTST --lr 0.0001 --weight_decay 0.0 --epoches 100 --batchsize 128 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method PatchTST --lr 0.0001 --weight_decay 0.0 --epoches 100 --batchsize 128 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method PatchTST --lr 0.0001 --weight_decay 0.0 --epoches 100 --batchsize 128 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method Timesnet --lr 0.001 --weight_decay 0.0000001 --epoches 10 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method Timesnet --lr 0.001 --weight_decay 0.0000001 --epoches 10 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method Timesnet --lr 0.001 --weight_decay 0.0000001 --epoches 10 --batchsize 256 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method Timesnet --lr 0.001 --weight_decay 0.0000001 --epoches 10 --batchsize 256 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method tsmixer --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method tsmixer --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method tsmixer --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method tsmixer --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025

  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method GPT4TS --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method GPT4TS --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method GPT4TS --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method GPT4TS --lr 0.0001 --weight_decay 0.0 --epoches 10 --batchsize 64 --seed 2025


  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1D  --method stemGNN --lr 0.0001 --weight_decay 0.0 --epoches 50 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 3D  --method stemGNN --lr 0.0001 --weight_decay 0.0 --epoches 50 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 5D  --method stemGNN --lr 0.0001 --weight_decay 0.0 --epoches 50 --batchsize 32 --seed 2025
  python ./src/main_three_parts.py --dataset_path=./dataset/Processed_hour --dataset $name --length_input 2D --length_output 1W  --method stemGNN --lr 0.0001 --weight_decay 0.0 --epoches 50 --batchsize 32 --seed 2025

done



