python -m tools.export_onnx \
    --cfg_path config/nanodet-plus-m_320_exam.yml \
    --model_path weights/nanodet_plus_m_320.pth \
    --out_path weights/nanodet_plus_m_320_full.onnx \
    --input_shape 320,320 \
    --opset 12
python tools\remove_initializer_from_input.py \
    --input weights/nanodet_plus_m_320_full.onnx \
    --output weights/nanodet_plus_m_320.onnx