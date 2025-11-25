# ECS289A-final-project
ECS289A Final project for Nathan Kotni, Qikai Ma, Yinuo Tang, Xingming Xu.

# Setup

Ideally, use Google Colab with access to L4/A100 GPUs to run this project (estimated time: 30 mins on A100 GPU for full fine tuning + eval on A100). 

The details to run each notebook, baseline.ipynb, and with_lora.ipynb, are both contained in the respective notebooks.

Baseline.ipynb is 

Additionally, [filtered_dataset.json](https://github.com/xingmingxu/ECS289A-final-project/blob/main/filtered_dataset.json) is our curated dataset, obtained from [Chart-to-text](https://github.com/vis-nlp/Chart-to-text/tree/main/statista_dataset/dataset) which we have repurposed for our project. The script that we used to obtain that json file is located at filterCaptions.py for your reference.

# Contact
If you have any questions about this work, please contact **Xingming Xu** using the following email address: xmxu at ucdavis.edu (other authors please add your info here). 

# Models & Evaluation

Please see our paper.

# References

We used the following dataset:

@inproceedings{kantharaj-etal-2022-chart,
    title = "Chart-to-Text: A Large-Scale Benchmark for Chart Summarization",
    author = "Kantharaj, Shankar  and
      Leong, Rixie Tiffany  and
      Lin, Xiang  and
      Masry, Ahmed  and
      Thakkar, Megh  and
      Hoque, Enamul  and
      Joty, Shafiq",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.277",
    doi = "10.18653/v1/2022.acl-long.277",
    pages = "4005--4023",
}
