### Original Installation

```bash
conda create --name wilor python=3.10
conda activate wilor

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# Install requirements
pip install -r requirements.txt
```
Download the pretrained models using: 
```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/` folder. 
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).
## Demo
```bash
python demo.py --img_folder demo_img --out_folder demo_out --save_mesh 
```

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [HaMeR](https://github.com/geopavlakos/hamer/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## License 
WiLoR models fall under the [CC-BY-NC--ND License](./license.txt). This repository depends also on [Ultralytics library](https://github.com/ultralytics/ultralytics) and [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses. By using this repository, you must also comply with the terms of these external licenses.
## Citing
If you find WiLoR useful for your research, please consider citing our paper:

```bibtex
@misc{potamias2024wilor,
    title={WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild},
    author={Rolandos Alexandros Potamias and Jinglei Zhang and Jiankang Deng and Stefanos Zafeiriou},
    year={2024},
    eprint={2409.12259},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
