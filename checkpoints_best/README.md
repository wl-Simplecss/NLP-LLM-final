## checkpoints_best 说明

该目录用于存放“效果最好的三类模型 checkpoint”，方便直接用 `inference.py` 做一键推理。

### 当前包含

- **RNN 最优**：`rnn_best.pt`  
  来源：`checkpoints/checkpoints_rnn_tf_teacher_forcing/rnn_epoch18.pt`（在统一测试集 BLEU 表中 RNN 组 Beam BLEU 最优）
- **Transformer 最优**：`transformer_best.pt`  
  来源：`checkpoints/checkpoints_transformer_pos_relative/transformer_epoch58.pt`（Transformer 组 BLEU 最优）
- **T5 最优**：`t5_best.pt`  
  来源：`checkpoints/checkpoints_t5/t5_epoch10.pt`



