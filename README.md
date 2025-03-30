# MLflow End-to-End Pipeline with Best Practices

## 📌 Project Overview
This project demonstrates a **Machine Learning pipeline** with **MLflow** for tracking experiments and model training. It follows industry best practices and includes:

- **Data Loading**: Using `sklearn.datasets`
- **Data Preprocessing**: Feature selection and splitting
- **Model Training**: Random Forest with hyperparameters
- **Model Evaluation**: Performance metrics logged to MLflow
- **MLflow Integration**: Tracking and visualizing experiments

---

## 🚀 Getting Started
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/mlflow-pipeline.git
cd mlflow-pipeline
```

### **2️⃣ Set Up the Conda Environment**
Ensure you have **Miniconda** or **Anaconda** installed. Then run:
```bash
conda env create -f environment.yml
conda activate mlflow-pipeline
```

### **3️⃣ Run the Pipeline**
```bash
python src/run_pipeline.py --n_estimators 150 --max_depth 5 --test_size 0.2
```

### **4️⃣ Start MLflow UI** *(Optional but Recommended)*
```bash
mlflow ui
```
Then open **http://localhost:5000** to view experiment logs.

---

## 📂 Project Structure
```
mlflow_pipeline/
│── data/               # Dataset (if applicable)
│── src/
│   ├── data_loader.py  # Load dataset from sklearn
│   ├── preprocessing.py # Data processing & splitting
│   ├── train.py        # Model training script
│   ├── evaluate.py     # Model evaluation metrics
│   ├── run_pipeline.py # Main pipeline script
│── tests/              # Unit tests
│── environment.yml     # Conda dependencies
│── README.md           # Project documentation
```

---

## 🔍 Example Output
```
Model logged with accuracy: 0.95
Run `mlflow ui` to view experiment logs.
```

---

## 📢 Contributing
Feel free to **fork** this repository, submit **issues**, or create **pull requests**.

---

## 📝 License
This project is open-source and available under the **MIT License**.

---

💡 **Showcase this to your interviewer by explaining:**
- How MLflow tracks experiments
- Why modularized code improves maintainability
- How hyperparameter tuning improves model performance

🚀 **Good luck with your interview!**

