** PROJECT DIRECTORY STRUCTURE **

Crop_Stress_Detection_Using_Contrastive_Learning_and_XAI/
├── data/
│   ├── csv_data/
│   │   ├── patch_meta.csv
│   │   └── stress_cluster_labels.csv
│   ├── embeddings/
│   ├── multi_data/
│   │   ├── indices_jan_2023.tif
│   │   ├── indices_feb_2023.tif
│   │   ├── indices_mar_2023.tif
│   │   ├── indices_apr_2023.tif
│   │   ├── indices_may_2023.tif
│   │   └── indices_jun_2023.tif
│   ├── patches/
│   ├── remain_data/
|   |   └── indices files not used .tif
│   ├── series/
|   └── 2023_gee_script.js
│
├── models/
│   ├── encoder_simclr.pt
│   └── stress_classifier.pkl
│
├── new_data/
│   ├── 2024_multi_data/
│   │   ├── multi_mar_2024.tif
│   │   ├── multi_apr_2024.tif
│   │   ├── multi_may_2024.tif
│   │   ├── multi_oct_2024.tif
│   │   └── multi_dec_2024.tif
│   ├── 2024_remain_data/
|       └── indices files not used .tif
│   ├── new_patches/
│   ├── new_series/
|   ├── output_predictions/
│   └── 2024_gee_script.js
│
├── outputs/
│   ├── shap_explanations/
|   |   ├── <month>_force.png
│   │   └── <month>_summary.png
│   ├── label_pca_scatter.png
│   └── model_accuracy_comparison.png
│
├── src/
│   ├── __pycache__/
│   ├── catboost_info/
│   ├── redundant/   ---- all files as already there
│   ├── testing/
│   │   ├── __pycache__/
│   │   ├── generate_patches_test.py
│   │   ├── predict_stress.py
│   │   ├── shap_explain.py
│   │   └── timeseries_test.py
│   └── training/
│       ├── __pycache__/
│       ├── create_patches.py
│       ├── dataset.py
│       ├── extract_embedding.py
│       ├── generate_labels.py
│       ├── model_comparison.ipynb
│       ├── model.py
│       ├── timeseries.py
│       ├── train_classifier.py
│       └── train_simclr.py
│
├── .gitignore
├── how_to_run.txt
└── requirements.txt
