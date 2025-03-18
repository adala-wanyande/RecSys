# **Assignment 1: Neural Collaborative Filtering on the MovieLens 1M Dataset**

## **Data Preprocessing**
The preprocessing pipeline is implemented in [`data_preprocessing.py`](./data_preprocessing.py). This script prepares the **MovieLens 1M dataset** by performing the following steps:

1. **Load the dataset**  
   - Reads `movies.dat`, `ratings.dat`, and `users.dat` using `pandas`.
   - Handles special characters and ensures proper parsing using `sep="::"`.

2. **Convert explicit ratings to implicit feedback**  
   - Ratings **≥ 4** are labeled as **positive interactions (1)**.
   - Lower ratings are ignored, as the study is based on implicit feedback.

3. **Generate negative samples**  
   - Negative interactions (**0-label**) are sampled from movies the user **never rated**.

4. **Shuffle and split the dataset**  
   - The data is randomly shuffled and split into:
     - **70% Training**
     - **15% Validation**
     - **15% Testing**

5. **Save processed data**  
   - The final datasets are saved as:
     - `./data/train_data.csv`
     - `./data/val_data.csv`
     - `./data/test_data.csv`

### **Running the Preprocessing Script**
To generate the processed dataset, run:
```bash
python data_preprocessing.py
```
This will output dataset statistics and store the files in the `./data/` directory.

