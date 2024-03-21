# CITB: A Benchmark for Continual Instruction Tuning

This repository includes the data and code of the paper: [CITB: A Benchmark for Continual Instruction Tuning](https://arxiv.org/abs/2310.14510) (**Findings of EMNLP 2023**) by *Zihan Zhang*, *Meng Fang*, *Ling Chen*, and *Mohammad-Reza Namazi-Rad*.


- [⚙️ Install Dependencies](#️-install-dependencies)
- [📋 Data](#-data)
- [📊 Reproduce the Results](#-reproduce-the-results)
  - [Run Initial Multi-task Fine-tuning (Stage 1)](#run-initial-multi-task-fine-tuning-stage-1)
  - [Run Sequential Single Task Fine-tuning (Stage 2)](#run-sequential-single-task-fine-tuning-stage-2)
  - [Run Ablation Studies](#run-ablation-studies)
  - [Collect Evaluation Results](#collect-evaluation-results)
- [🌟Citation](#citation)
- [👏Acknowledgement](#acknowledgement)
- [🐞Questions?](#questions)



## ⚙️ Install Dependencies

The code has been tested under Python 3.9. The following are the steps to set up the environment.

Create conda environment:
```bash
conda create -n citb python=3.9 -y
conda activate citb
```

Install [PyTorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-24): we used Pytorch 1.10.0 and CUDA 11.3 in the experiment; however, other versions might also work.
```bash
# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Install libraries:
```bash
pip install -r requirements.txt
```

## 📋 Data 

We use the instruction data from [Super-NaturalInstructions](https://arxiv.org/abs/2204.07705). The processed data for the tasks in the **InstraDialog** and **InstraDialog++** streams are available in the [`data/`](https://github.com/hyintell/CITB/tree/main/data) folder. We also provide the scripts to split tasks under the [`scripts/data_scripts/`](https://github.com/hyintell/CITB/tree/main/scripts/data_scripts) folder.

- The **InstraDialog** stream has 19 tasks, which are all dialogue-related tasks, including 4 tasks from dialogue state tracking, 11 tasks from dialogue generation, and 4 tasks from intent identification.
- The **InstraDialog++** stream has 38 tasks, including all 19 tasks from the **InstraDialog** stream and  19 other tasks from broad categories, including sentence ordering, style transfer, toxic language detection, etc.


## 📊 Reproduce the Results

We have provided executable scripts to reproduce the results. Refer to the `.sh` files for different settings under the [`scripts/`](https://github.com/hyintell/CITB/tree/main/scripts) folder. We provide our results under the [`scores/`](https://github.com/hyintell/CITB/tree/main/scores) folder.

### Run Initial Multi-task Fine-tuning (Stage 1)

Train an initial model for better following instructions (`Init` baseline):
```bash
bash run_initial_multitask_tuning.sh
```

Joint train an initial model with the subsequent tasks (`Multi` baseline):
```bash
bash run_initial_multitask_tuning_with_CL.sh
```

### Run Sequential Single Task Fine-tuning (Stage 2)

Run different CL baselines for the **InstraDialog** stream:
```bash
bash short_stream_scripts/meta_job.sh
```

Run different CL baselines for the **InstraDialog++** stream:
```bash
bash long_stream_scripts/meta_job.sh
```

### Run Ablation Studies

```bash
bash ablation/{xxx}.sh
```

### Collect Evaluation Results

```bash
bash score_scripts/{xxx}.sh
```


> [!NOTE]  
> In the experiments, we used T5 as the base LM; you may choose other models from HuggingFace (such as instruction-finetuned models); however, you may need to change the CL code accordingly.



## 🌟Citation

If you find our code, data, or the paper useful, please cite the paper:

```bibtex
@inproceedings{zhang-etal-2023-citb,
    title = "{CITB}: A Benchmark for Continual Instruction Tuning",
    author = "Zhang, Zihan  and
      Fang, Meng  and
      Chen, Ling  and
      Namazi-Rad, Mohammad-Reza",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.633",
    doi = "10.18653/v1/2023.findings-emnlp.633",
    pages = "9443--9455",
    abstract = "Continual learning (CL) is a paradigm that aims to replicate the human ability to learn and accumulate knowledge continually without forgetting previous knowledge and transferring it to new tasks. Recent instruction tuning (IT) involves fine-tuning models to make them more adaptable to solving NLP tasks in general. However, it is still uncertain how instruction tuning works in the context of CL tasks. This challenging yet practical problem is formulated as Continual Instruction Tuning (CIT). In this work, we establish a CIT benchmark consisting of learning and evaluation protocols. We curate two long dialogue task streams of different types, InstrDialog and InstrDialog++, to study various CL methods systematically. Our experiments show that existing CL methods do not effectively leverage the rich natural language instructions, and fine-tuning an instruction-tuned model sequentially can yield similar or better results. We further explore different aspects that might affect the learning of CIT. We hope this benchmark will facilitate more research in this direction.",
}
```

## 👏Acknowledgement

Our data and code are based on previous works:
- [Super-NaturalInstructions](https://github.com/allenai/natural-instructions)
- [Tk-Instruct](https://github.com/yizhongw/Tk-Instruct)


## 🐞Questions?
If you have questions, please raise an [issue](https://github.com/hyintell/RetrievalQA/issues). 
