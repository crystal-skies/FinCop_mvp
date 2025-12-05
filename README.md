# FinCop_mvp
Hi! This is a small mvp to start showing what we have. 

## Prerequisites
1. Drivers of NVIDIA (`nvidia-smi`)
2. Conda  

## Instalation
We're using python=3.9 for compatibility with Deep Learning.

```bash
conda create --name ocr python=3.9 -y
conda activate ocr
```

## Install Libraries of NVIDIA
```bash
conda install cudatoolkit=11.8 cudnn -c conda-forge -y
```

## Install PaddlePaddle (GPU)
Versión 2.6.1 compatible with Cuda 11.x.

```bash
python -m pip install paddlepaddle-gpu==2.6.1
```

## Install PaddleOCR and Dependencies
Those are: { paddleocr==2.7.3, numpy==1.24.3, opencv...<4.10, protobuf==3.20.3} 
```bash
pip install paddleocr==2.7.3 protobuf==3.20.3 "numpy==1.24.3" "opencv-python-headless<4.10" streamlit
```

## Additional Configuration 
Just in case, sometimes the system will not find the libraryes cudnn even when are installed. Thats
why you should export it. 
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```
## Execution
Then Execute.
```bash
streamlit run app.py
```

## Care with this 
You should take care about the initializers. 
```python
# app.py
ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='es', 
    use_gpu=True,          # Activa GPU
    ocr_version='PP-OCRv3' # Usar v3 (v4 aún no tiene pesos oficiales 'es' en PyPI)
)
```


# Seahorse <- para solucionar el problema con las credenciales. 