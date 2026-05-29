
# Downstream Datasets Setup Guide

This guide provides instructions on how to download, extract, and structure the benchmark datasets used in the oncology downstream evaluation tasks.

---

## Dataset Directory Structure

To ensure the evaluation scripts can locate the data correctly, your directory structure under `downstreams_demo/downstreams_demo/` should look like this:

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
