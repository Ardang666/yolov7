#!/bin/bash
PROJ_NAME='yolov7'
ONNX_MODEL_PATH="../mm_onnx_models"
ONNX_MODEL=${ONNX_MODEL_PATH}/${PROJ_NAME}.onnx
MM_MODEL_PATH="../mm_magicmind_models"

CMD='/usr/local/neuware/samples/magicmind/mm_build/build/onnx_build'

# float32的模型生成过程
QUANT_MODE=force_float32
echo ${QUANT_MODE}
${CMD} --onnx ${ONNX_MODEL} --precision ${QUANT_MODE} --build_config config.json --magicmind_model ${MM_MODEL_PATH}/${PROJ_NAME}.${QUANT_MODE}.magicmind

# float16的模型生成过程
QUANT_MODE=force_float16
echo ${QUANT_MODE}
${CMD} --onnx ${ONNX_MODEL} --precision ${QUANT_MODE} --build_config config.json --magicmind_model ${MM_MODEL_PATH}/${PROJ_NAME}.${QUANT_MODE}.magicmind

# int8 量化模型的生成过程，这里使用公开数据集；根据需要指向自己的数据集
DATASETS_PATH=/public/datasets/COCO/val2017
DATASETS_PATH=/works/ynyd/02_pic_quality/app/test/images

QUANT_MODE=qint8_mixed_float16
echo ${QUANT_MODE}
python gen.py --onnx_model ${ONNX_MODEL} --shape_mutable true --quant_mode ${QUANT_MODE} --output ${MM_MODEL_PATH}/${PROJ_NAME}.${QUANT_MODE}.magicmind --image_dir ${DATASETS_PATH} 

