# Shakespear - GPT
This project contains a GPT-2 model trained on Julius Caesar Play written by Shakespear. GPT-2 is a transformer-based language model that can generate human-like text. The model architecture is heavily based on the Andrey Karpathy's video: "[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)"

Model has been trained on Kaggle using the 16GB - T4 GPU. Maximum batch size that can be used without getting OOM error was 4. The model has been trained for 60 epochs.

## Model Highlights
- This GPT model has 12 transformer blocks
- Total number of parameters in the model is 85.9 M
- Tokenizer used here is a manually created text to char encoder
- Vocab size is 65
- In the self-attention block, masking is used to prevent the model from looking into the future tokens


## HuggingFace Space
The model is available on HuggingFace space. You can access the model using the following link: [Julius-Caesar-Play-GPT]()

## Model Architecture
```
======================================================================================================================================================
Layer (type (var_name))                            Input Shape               Output Shape              Param #                   Mult-Adds
======================================================================================================================================================
GPT (GPT)                                          [4, 1024]                 [4, 1024, 65]             --                        --
├─ModuleDict (transformer)                         --                        --                        --                        --
│    └─Embedding (wte)                             [4, 1024]                 [4, 1024, 768]            49,920                    199,680
│    └─Embedding (wpe)                             [1, 1024]                 [1, 1024, 768]            786,432                   786,432
│    └─Dropout (drop)                              [4, 1024, 768]            [4, 1024, 768]            --                        --
│    └─ModuleList (h)                              --                        --                        --                        --
│    │    └─TransformerBlock (0)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (1)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (2)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (3)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (4)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (5)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (6)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (7)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (8)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (9)                   [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (10)                  [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    │    └─TransformerBlock (11)                  [4, 1024, 768]            [4, 1024, 768]            --                        --
│    │    │    └─LayerNorm (ln_1)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─MultiHeadAttention (attn)         [4, 1024, 768]            [4, 1024, 768]            2,362,368                 9,449,472
│    │    │    └─LayerNorm (ln_2)                  [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
│    │    │    └─FeedForward (mlp)                 [4, 1024, 768]            [4, 1024, 768]            4,722,432                 18,889,728
│    └─LayerNorm (ln_f)                            [4, 1024, 768]            [4, 1024, 768]            1,536                     6,144
├─Linear (lm_head)                                 [4, 1024, 768]            [4, 1024, 65]             49,920                    199,680
======================================================================================================================================================
Total params: 85,942,272
Trainable params: 85,942,272
Non-trainable params: 0
Total mult-adds (M): 341.41
======================================================================================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 3380.64
Params size (MB): 343.77
Estimated Total Size (MB): 3724.44
======================================================================================================================================================
```

## Training Logs
```
2025-01-14 19:20:45 | INFO | Logging setup complete. Logs will be saved to: /kaggle/working/training_20250114_192045.log
2025-01-14 19:23:42 | INFO | Epoch 1/150, Average Loss: 2.7452
2025-01-14 19:26:38 | INFO | Epoch 2/150, Average Loss: 2.4883
2025-01-14 19:29:35 | INFO | Epoch 3/150, Average Loss: 2.4743
2025-01-14 19:32:32 | INFO | Epoch 4/150, Average Loss: 2.4590
2025-01-14 19:35:29 | INFO | Epoch 5/150, Average Loss: 2.4399
2025-01-14 19:38:27 | INFO | Epoch 6/150, Average Loss: 2.3914
2025-01-14 19:41:23 | INFO | Epoch 7/150, Average Loss: 2.2681
2025-01-14 19:44:20 | INFO | Epoch 8/150, Average Loss: 2.0732
2025-01-14 19:47:17 | INFO | Epoch 9/150, Average Loss: 1.8946
2025-01-14 19:50:14 | INFO | Epoch 10/150, Average Loss: 1.7543
2025-01-14 19:53:11 | INFO | Epoch 11/150, Average Loss: 1.6499
2025-01-14 19:56:08 | INFO | Epoch 12/150, Average Loss: 1.5720
2025-01-14 19:59:05 | INFO | Epoch 13/150, Average Loss: 1.5124
2025-01-14 20:02:02 | INFO | Epoch 14/150, Average Loss: 1.4672
2025-01-14 20:04:59 | INFO | Epoch 15/150, Average Loss: 1.4202
2025-01-14 20:07:56 | INFO | Epoch 16/150, Average Loss: 1.3859
2025-01-14 20:10:53 | INFO | Epoch 17/150, Average Loss: 1.3521
2025-01-14 20:13:50 | INFO | Epoch 18/150, Average Loss: 1.3270
2025-01-14 20:16:47 | INFO | Epoch 19/150, Average Loss: 1.2937
2025-01-14 20:19:44 | INFO | Epoch 20/150, Average Loss: 1.2778
2025-01-14 20:22:42 | INFO | Epoch 21/150, Average Loss: 1.2548
2025-01-14 20:25:39 | INFO | Epoch 22/150, Average Loss: 1.2245
2025-01-14 20:28:36 | INFO | Epoch 23/150, Average Loss: 1.2003
2025-01-14 20:31:33 | INFO | Epoch 24/150, Average Loss: 1.1842
2025-01-14 20:34:30 | INFO | Epoch 25/150, Average Loss: 1.1657
2025-01-14 20:37:28 | INFO | Epoch 26/150, Average Loss: 1.1301
2025-01-14 20:40:25 | INFO | Epoch 27/150, Average Loss: 1.1127
2025-01-14 20:43:22 | INFO | Epoch 28/150, Average Loss: 1.0946
2025-01-14 20:46:19 | INFO | Epoch 29/150, Average Loss: 1.0687
2025-01-14 20:49:16 | INFO | Epoch 30/150, Average Loss: 1.0432

2025-01-15 03:45:09 | INFO | Training resumed from the last checkpoint: epoch 30
2025-01-15 03:47:45 | INFO | Epoch 31/150, Average Loss: 1.0322
2025-01-15 03:50:28 | INFO | Epoch 32/150, Average Loss: 1.0340
2025-01-15 03:53:14 | INFO | Epoch 33/150, Average Loss: 1.0294
2025-01-15 03:56:00 | INFO | Epoch 34/150, Average Loss: 1.0281
2025-01-15 03:58:46 | INFO | Epoch 35/150, Average Loss: 1.0334
2025-01-15 04:01:32 | INFO | Epoch 36/150, Average Loss: 1.0341
2025-01-15 04:04:18 | INFO | Epoch 37/150, Average Loss: 1.0374
2025-01-15 04:07:05 | INFO | Epoch 38/150, Average Loss: 1.0347
2025-01-15 04:09:51 | INFO | Epoch 39/150, Average Loss: 1.0335
2025-01-15 04:12:37 | INFO | Epoch 40/150, Average Loss: 1.0342
2025-01-15 04:15:23 | INFO | Epoch 41/150, Average Loss: 1.0326
2025-01-15 04:18:09 | INFO | Epoch 42/150, Average Loss: 1.0291
2025-01-15 04:20:55 | INFO | Epoch 43/150, Average Loss: 1.0327
2025-01-15 04:23:41 | INFO | Epoch 44/150, Average Loss: 1.0323
2025-01-15 04:26:27 | INFO | Epoch 45/150, Average Loss: 1.0279
2025-01-15 04:29:13 | INFO | Epoch 46/150, Average Loss: 1.0252
2025-01-15 04:31:59 | INFO | Epoch 47/150, Average Loss: 1.0282
2025-01-15 04:34:45 | INFO | Epoch 48/150, Average Loss: 1.0328
2025-01-15 04:37:31 | INFO | Epoch 49/150, Average Loss: 1.0335
2025-01-15 04:40:17 | INFO | Epoch 50/150, Average Loss: 1.0358
2025-01-15 04:43:03 | INFO | Epoch 51/150, Average Loss: 1.0296
2025-01-15 04:45:49 | INFO | Epoch 52/150, Average Loss: 1.0277

2025-01-15 18:05:35 | INFO | Training resumed from the last checkpoint: epoch 52
2025-01-15 18:08:28 | INFO | Epoch 53/150, Average Loss: 1.0284
2025-01-15 18:11:17 | INFO | Epoch 54/150, Average Loss: 1.0279
2025-01-15 18:14:06 | INFO | Epoch 55/150, Average Loss: 1.0334
2025-01-15 18:16:55 | INFO | Epoch 56/150, Average Loss: 1.0368
2025-01-15 18:19:44 | INFO | Epoch 57/150, Average Loss: 1.0380
2025-01-15 18:22:33 | INFO | Epoch 58/150, Average Loss: 1.0267
2025-01-15 18:25:22 | INFO | Epoch 59/150, Average Loss: 1.0348
2025-01-15 18:28:10 | INFO | Epoch 60/150, Average Loss: 1.0303

```


