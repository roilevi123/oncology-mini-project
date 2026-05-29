# the place of the code 
in the folder 

# Downstream Datasets Setup Guide & Data Structure

This guide provides detailed instructions for downloading, extracting, and organizing the required datasets, along with an explanation of the internal data structure and architecture of each downstream task.

---

# 📂 Directory Structure

To ensure that the evaluation scripts and data loaders can locate the datasets correctly, extract and organize all datasets inside the `downstreams_demo/downstreams_demo/` directory exactly as shown below:

```text
downstreams_demo/downstreams_demo/
├── LC25000/
│   └── lung_colon_image_set/
│       ├── colon_image_sets/
│       └── lung_image_sets/
├── pannuke/
│   ├── Images/
│   └── Masks/
├── PathMMU/
│   ├── images/
│   └── pathmmu_test.json
├── skincancer/
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   └── HAM10000_metadata.csv
└── unitopatho/
    ├── 800m/
    └── 800m_0.json
```

> Ensure that the folder names and file locations match this structure exactly. Incorrect placement may cause the data loaders to fail.

---

# 🔬 Dataset Sources and Internal Structure

## 1. LC25000 (LC25K)

**Download Source:**
Kaggle – *Lung and Colon Cancer Histopathological Images*

### Dataset Structure

This dataset does not rely on external metadata files such as JSON or CSV. Instead, it follows a standard image-folder classification structure.

The data loader automatically scans the subdirectories and uses the folder names as class labels:

| Folder                        | Label                        |
| ----------------------------- | ---------------------------- |
| `colon_image_sets/colon_aca/` | Colon adenocarcinoma         |
| `colon_image_sets/colon_n/`   | Normal colon tissue          |
| `lung_image_sets/lung_aca/`   | Lung adenocarcinoma          |
| `lung_image_sets/lung_scc/`   | Lung squamous cell carcinoma |
| `lung_image_sets/lung_n/`     | Normal lung tissue           |

---

## 2. PanNuke

**Download Source:**
Official University of Warwick release or Kaggle – *PanNuke Dataset*

### Dataset Structure

PanNuke is designed for nucleus-level instance segmentation and classification.

The dataset consists of two synchronized components:

### `Images/`

Contains tissue images from multiple cancer types.

### `Masks/`

Contains segmentation masks corresponding to the images in the `Images/` directory.

Each pixel in a mask represents:

* Whether it belongs to a cell nucleus.
* The category of the nucleus (e.g., epithelial, inflammatory, connective tissue, neoplastic, etc.).

The image and mask files must remain aligned and maintain matching dimensions.

---

## 3. PathMMU

**Download Source:**
Hugging Face Datasets (search for the dataset used by the retrieval benchmark and referenced by the data loader).

### Dataset Structure

PathMMU is a multimodal pathology retrieval benchmark linking pathology images with free-text medical descriptions.

### `pathmmu_test.json`

This JSON file serves as the core metadata file and contains entries with:

* The image path inside the `images/` directory.
* The corresponding clinical description or caption.

The model uses these image-text pairs for cross-modal retrieval tasks by comparing image embeddings with text embeddings.

### `images/`

Contains the pathology images referenced by the JSON metadata file.

---

## 4. Skin Cancer (HAM10000)

**Download Source:**
Harvard Dataverse (official release) or Kaggle – *HAM10000 Skin Cancer Dataset*

### Dataset Structure

This dataset relies on a centralized CSV metadata file.

### `HAM10000_metadata.csv`

Contains important fields such as:

* `lesion_id`
* `image_id`
* `dx` (diagnostic label)

Examples of diagnosis labels include:

* `mel` – Melanoma
* `nv` – Melanocytic nevus
* `bcc` – Basal cell carcinoma

### `HAM10000_images_part_1/`

### `HAM10000_images_part_2/`

Contain the raw dermoscopic images.

The data loader reads each metadata row, locates the corresponding image using the `image_id`, and assigns the diagnostic label specified in the CSV file.

---

## 5. UniToPatho

**Download Source:**
IEEE Dataport (official release) or Kaggle – *UniToPatho*

### Dataset Structure

UniToPatho is a histopathology benchmark for colorectal lesion classification based on the Vienna Classification system.

### `800m/`

Contains image patches extracted at a specific microscope magnification level.

### `800m_0.json`

Metadata file that maps each image to its corresponding lesion category.

Typical categories include:

* Normal tissue
* Adenoma
* High-grade dysplasia
* Invasive carcinoma

The JSON file is used by the data loader to associate image samples with their ground-truth labels.

---

# ✅ Final Checklist

Before running the downstream evaluation pipeline, verify that:

* All datasets have been downloaded and extracted.
* Folder names exactly match the expected names.
* Metadata files (`.json`, `.csv`) are located in the correct directories.
* Image folders contain the complete extracted dataset contents.
* The directory structure matches the layout shown above.

Following this setup will ensure compatibility with the provided data loaders and evaluation scripts.
