# Internvl2-8BDetect
finetune internvl2-8B to detect AI-generated images/videos by LoRA.

## 🛠️ Requirements and Installation

### 🏋️‍♂Installation via Pip

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

## 🏋️‍♂️ Prepare Model

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
    ```jsonL
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
- `GPUS`: How many GPUS you wish to use
- `BATCH_SIZE`: batch size in training process
- `PER_DEVICE_BATCH_SIZE`: batch size in each GPU
- `GRADIENT_ACC`: Cumulative number of steps in a gradient
- `OUTPUT_DIR`: Path for saving the finetuned model

Modify these variables as needed to adapt the training process to different datasets and setups.

##  📚Test

You can quickly run the demo script by executing:

```python
python llm_test/inter_test.py
```
Or you can use the model by flask to avoid loading for each time.

```python
python llm_test/single_video_chat.py
python llm_test/request.py
```

