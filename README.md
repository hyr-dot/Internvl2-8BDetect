# Internvl2-8BDetect
finetune internvl2-8B to detect AI-generated images/videos by LoRA.

## 🛠️ Requirements and Installation

### Installation via Pip

1. Ensure your environment meets the following requirements:
    - Python == 3.9
    - Pytorch == 2.4.0
    - CUDA Version == 12.1

2. Clone the repository:
    ```bash
    git clone https://github.com/hyr-dot/Internvl2-8BDetect.git
    cd Internvl2-8BDetect
    ```
3. Install dependencies:
    ```bash
    apt update && apt install git
    pip install -r requirements.txt
    ```

## 🤖 Prepare Model

1. **Download Internvl2-8B weights from Hugging Face**
   
   You can download the original Internvl2-8B weights from https://huggingface.co/OpenGVLab/InternVL2-8B.

2. **Ensure the files are placed correctly**
   
   Organize your `Internvl2-8BDetect/` folder as follows:
   ```
    Internvl2-8BDetect/
    ├── Internvl2-8B/
    ├── data/
    ├── finetune/
    ├── internvl/
    ├── llm_test/
   
   ```
## 🤖 Prepare Customized Data

After downloading the pre-trained model, prepare your customized SFT (Supervised Fine-Tuning) data. Create a JSON file in Internvl2-8BDetect/data/ similar to the example below.

The format for the JSON file should be:
    ```json
{
  "your-custom-dataset-1": {
    "root": "path/to/the/image/",
    "annotation": "path/to/the/jsonl/annotation",
    "data_augment": false,
    "max_dynamic_patch": 12,
    "repeat_time": 1,
    "length": "number of samples in the dataset"
  }
}
    ```
You can refer to Internvl2-8BDetect/data/941split.json for example.

For each data item in the "root" folder, there must be a corresponding entry in the file specified by the "annotation" field. A concrete example can be found in Internvl2-8BDetect/data/data_example.jsonl.

The format for each specific JSONL (such as plain text data, single-image data, multi-image data, video data) can be organized according to the descriptions provided in [this document](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html).
My suggestion is to add new domain-specific data on top of the general data from our open-sourced InternVL 1.2. This will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

## 🚀LoRA Finetune InternVL2-8B

You can fine-tune internvl2 using LoRA with the following script:

```bash
bash ./finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh
```

The script allows customization through the following environment variables:
- `OUTPUT_DIR`: Directory for saving training output
- `DATA_PATH`: Path to the training dataset
- `WEIGHT_PATH`: Path to the pre-trained weights
- `TRAIN_DATA_CHOICE`: Selecting the training dataset
- `VAL_DATA_CHOICE`: Selecting the validation dataset

Modify these variables as needed to adapt the training process to different datasets and setups.

##  Quick Start

### CLI Demo

You can quickly run the demo script by executing:

```bash
bash scripts/cli_demo.sh
```

The `cli_demo.sh` script allows customization through the following environment variables:
- `WEIGHT_PATH`: Path to the FakeShield weight directory (default: `./weight/fakeshield-v1-22b`)
- `IMAGE_PATH`: Path to the input image (default: `./playground/image/Sp_D_CRN_A_ani0043_ani0041_0373.jpg`)
- `DTE_FDM_OUTPUT`: Path for saving the DTE-FDM output (default: `./playground/DTE-FDM_output.jsonl`)
- `MFLM_OUTPUT`: Path for saving the MFLM output (default: `./playground/DTE-FDM_output.jsonl`)

Modify these variables to suit different use cases.

## 🏋️‍♂️ Train

### Training Data Preparation

The training dataset consists of three types of data:

1. **PhotoShop Manipulation Dataset:** [CASIAv2](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset), [Fantastic Reality](http://zefirus.org/MAG)
2. **DeepFake Manipulation Dataset:** [FFHQ](https://cvlab.cse.msu.edu/dffd-dataset.html), [FaceAPP](https://cvlab.cse.msu.edu/dffd-dataset.html)
3. **AIGC-Editing Manipulation Dataset:** SD_inpaint Dataset (Coming soon)
4. **MMTD-Set Dataset:** MMTD-Set (Coming soon)


### Validation Data Preparation

The validation dataset consists of three types of data:

1. **PhotoShop Manipulation Dataset:** [CASIA1+](https://github.com/proteus1991/PSCC-Net?tab=readme-ov-file#testing), [IMD2020](http://zefirus.org/MAG), [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/), [coverage](https://github.com/wenbihan/coverage), [NIST16](https://mfc.nist.gov/), [DSO](https://recodbr.wordpress.com/code-n-data/#dso1_dsi1), [Korus](https://pkorus.pl/downloads/dataset-realistic-tampering)
2. **DeepFake Manipulation Dataset:** [FFHQ](https://cvlab.cse.msu.edu/dffd-dataset.html), [FaceAPP](https://cvlab.cse.msu.edu/dffd-dataset.html)
3. **AIGC-Editing Manipulation Dataset:** SD_inpaint Dataset (Coming soon)
4. **MMTD-Set Dataset:** MMTD-Set (Coming soon)

Download them from the above links and organize them as follows:

```bash
dataset/
├── photoshop/                # PhotoShop Manipulation Dataset
│   ├── CASIAv2_Tp/           # CASIAv2 Tampered Images
│   │   ├── image/
│   │   └── mask/
│   ├── CASIAv2_Au/           # CASIAv2 Authentic Images
│   │   └── image/
│   ├── FR_Tp/                # Fantastic Reality Tampered Images
│   │   ├── image/
│   │   └── mask/
│   ├── FR_Au/                # Fantastic Reality Authentic Images
│   │   └── image/
│   ├── CASIAv1+_Tp/          # CASIAv1+ Tampered Images
│   │   ├── image/
│   │   └── mask/
│   ├── CASIAv1+_Au/          # CASIAv1+ Authentic Images
│   │   └── image/
│   ├── IMD2020_Tp/           # IMD2020 Tampered Images
│   │   ├── image/
│   │   └── mask/
│   ├── IMD2020_Au/           # IMD2020 Authentic Images
│   │   └── image/
│   ├── Columbia/             # Columbia Dataset
│   │   ├── image/
│   │   └── mask/
│   ├── coverage/             # Coverage Dataset
│   │   ├── image/
│   │   └── mask/
│   ├── NIST16/               # NIST16 Dataset
│   │   ├── image/
│   │   └── mask/
│   ├── DSO/                  # DSO Dataset
│   │   ├── image/
│   │   └── mask/
│   └── Korus/                # Korus Dataset
│       ├── image/
│       └── mask/
│
├── deepfake/                 # DeepFake Manipulation Dataset
│   ├── FaceAPP_Train/        # FaceAPP Training Data
│   │   ├── image/
│   │   └── mask/
│   ├── FaceAPP_Val/          # FaceAPP Validation Data
│   │   ├── image/
│   │   └── mask/
│   ├── FFHQ_Train/           # FFHQ Training Data
│   │   └── image/
│   └── FFHQ_Val/             # FFHQ Validation Data
│       └── image/
│
├── aigc/                     # AIGC Editing Manipulation Dataset
│   ├── SD_inpaint_Train/     # Stable Diffusion Inpainting Training Data
│   │   ├── image/
│   │   └── mask/
│   ├── SD_inpaint_Val/       # Stable Diffusion Inpainting Validation Data
│   │   ├── image/
│   │   └── mask/
│   ├── COCO2017_Train/       # COCO2017 Training Data
│   │   └── image/
│   └── COCO2017_Val/         # COCO2017 Validation Data
│       └── image/
│
└── MMTD_Set/                 # Multi-Modal Tamper Description Dataset
    └── MMTD-Set-34k.json     # JSON Training File
```





### LoRA Finetune DTE-FDM

You can fine-tune DTE-FDM using LoRA with the following script:

```bash
bash ./scripts/DTE-FDM/finetune_lora.sh
```

The script allows customization through the following environment variables:
- `OUTPUT_DIR`: Directory for saving training output
- `DATA_PATH`: Path to the training dataset (JSON format)
- `WEIGHT_PATH`: Path to the pre-trained weights

Modify these variables as needed to adapt the training process to different datasets and setups.




## 🎯 Test

You can test FakeShield using the following script:

```bash
bash ./scripts/test.sh
```

The script allows customization through the following environment variables:

- `WEIGHT_PATH`: Path to the directory containing the FakeShield model weights.
- `QUESTION_PATH`: Path to the test dataset in JSONL format. This file can be generated using [`./playground/eval_jsonl.py`](https://github.com/zhipeixu/FakeShield/blob/main/playground/eval_jsonl.py).
- `DTE_FDM_OUTPUT`: Path for saving the output of the DTE-FDM model.
- `MFLM_OUTPUT`: Path for saving the output of the MFLM model.

Modify these variables as needed to adapt the evaluation process to different datasets and setups.

##  📚 Main Results

### Comparison of detection performance with advanced IFDL methods



## 📜 Citation

```bibtex
    @inproceedings{xu2024fakeshield,
            title={FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models},
            author={Xu, Zhipei and Zhang, Xuanyu and Li, Runyi and Tang, Zecheng and Huang, Qing and Zhang, Jian},
            booktitle={International Conference on Learning Representations},
            year={2025}
    }
```

## 🙏 Acknowledgement

We are thankful to LLaVA, groundingLMM, and LISA for releasing their models and code as open-source contributions.
