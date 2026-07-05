# 第一步

先在训练机上验证 model_7000.pt 能被正确加载和导出，不要先上 S100 板子。

```shell
# 在仓库根目录执行
conda activate extreme_parkour
cd legged_gym/legged_gym/scripts
python save_jit.py --exptid 020-00-distill --checkpoint 7000

# 成功后应生成类似:

legged_gym/logs/parkour_new/020-00-distill/traced/
├── 020-00-distill-7000-base_jit.pt
└── 020-00-distill-7000-vision_weight.pt
```

这一步的意义是确认三件事：

1. checkpoint 路径和权重是完整的。
2. depth_actor_state_dict、estimator_state_dict、depth_encoder_state_dict 能正常加载。
3. 当前 Python 环境能重建部署所需网络。

这一步通过后，第二步才是写 export_onnx.py，把模型拆成 depth_encoder.onnx 和 actor_estimator.onnx。

# 第二步

```shell
# 1.回到extreme-parkour根目录
cd ../../..

# 2.检查ONNX依赖
python -c "import onnx, onnxruntime; print('onnx ok')"

# 3.如果报错，先配置onnx环境；没有报错则略过该步骤
pip install onnx onnxruntime

# 4.新增一个导出脚本，例如：deploy/s100/export_onnx.py
#  这个脚本要从legged_gym/logs/parkour_new/020-00-distill/model_7000.pt  导出两个文件：
#  deploy/s100/export/depth_encoder.onnx
#  deploy/s100/export/actor_estimator.onnx
```