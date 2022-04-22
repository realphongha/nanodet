python -m tools.export_onnx \
    --cfg_path config/nanodet-plus-m_320_exam_iith_dmu.yml \
    --model_path workspace/nanodet-plus-m_320_exam_iith_dmu/model_best/nanodet_model_best.pth \
    --out_path workspace/nanodet-plus-m_320_exam_iith_dmu/nanodet_plus_m_320_coco_exam_iith_dmu_full.onnx \
    --input_shape 320,320 \
    --opset 12
python tools/remove_initializer_from_input.py \
    --input workspace/nanodet-plus-m_320_exam_iith_dmu/nanodet_plus_m_320_coco_exam_iith_dmu_full.onnx \
    --output workspace/nanodet-plus-m_320_exam_iith_dmu/nanodet_plus_m_320_coco_exam_iith_dmu.onnx
