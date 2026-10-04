# 🏠 California Housing Price Predictor — Deep Learning

A deep learning regression model built with **PyTorch** to predict California house prices using the **California Housing dataset** from Scikit-learn.

The project demonstrates a complete machine learning pipeline, including:

- Dataset loading and preprocessing
- Feature engineering
- Target transformation
- Feature standardization
- Custom PyTorch Dataset and DataLoader
- Multi-Layer Perceptron (MLP) architecture
- Model training with Adam optimizer
- Learning-rate scheduling
- Model evaluation using MAE, RMSE, and R²
- Prediction on unseen housing samples

---

## 📌 Project Overview

The goal of this project is to build a neural network capable of predicting the median house value of California districts based on demographic and housing-related features.

Instead of using a traditional machine learning regression algorithm, this project uses a **fully connected neural network (MLP)** implemented with PyTorch.

The model learns the relationship between housing characteristics and their corresponding median house values.

### Dataset

The project uses the **California Housing Dataset** provided by Scikit-learn:

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
```

The original dataset contains information about California housing districts, including:

- Median income
- Average house age
- Average number of rooms
- Average number of bedrooms
- Population
- Average occupancy
- Latitude
- Longitude

The original target variable is:

```text
MedHouseVal
```

which represents the median house value.

---

# 🧠 Model Architecture

The model is a fully connected **Multi-Layer Perceptron (MLP)** implemented using PyTorch.

The architecture is:

```text
Input Features
      │
      ▼
Linear(input_dim → 128)
      │
     SiLU
      │
      ▼
Linear(128 → 64)
      │
     SiLU
      │
      ▼
Linear(64 → 32)
      │
     SiLU
      │
      ▼
Linear(32 → 1)
      │
      ▼
Predicted House Value
```

### PyTorch Implementation

```python
class HousingMLP(nn.Module):
    def __init__(self, input_dim):
        super(HousingMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),

            nn.Linear(128, 64),
            nn.SiLU(),

            nn.Linear(64, 32),
            nn.SiLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
```

The model uses the **SiLU (Sigmoid Linear Unit)** activation function between the hidden layers.

---

# 🔄 Data Preprocessing

The preprocessing pipeline consists of several important steps.

## 1. Load the Dataset

The California Housing dataset is loaded as a Pandas DataFrame:

```python
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
```

---

## 2. Feature Engineering

Three additional features are created from the original dataset.

### Rooms Per Household

```python
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
```

This represents the average number of rooms relative to household occupancy.

### Bedrooms Per Room

```python
df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
```

This provides information about the proportion of bedrooms compared with total rooms.

### Population Per Household

```python
df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']
```

This provides another representation of population density.

These engineered features increase the input dimensionality from the original dataset and provide the neural network with additional relationships derived from the existing variables.

---

# 📉 Target Transformation

The target variable is transformed using a logarithmic transformation:

```python
y_log = np.log1p(df['MedHouseVal'].values).reshape(-1, 1)
```

The transformation is:

```text
log(1 + y)
```

This is useful for reducing the effect of skewed target values and making the regression problem easier for the neural network to learn.

After prediction, the transformation is reversed using:

```python
np.expm1(predictions)
```

---

# ✂️ Train/Test Split

The dataset is divided into training and testing sets using an 80/20 split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)
```

Therefore:

```text
80% → Training
20% → Testing
```

The `random_state=42` ensures reproducibility of the split.

---

# 📏 Feature Standardization

The input features are standardized using `StandardScaler`:

```python
scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
```

The scaler is fitted **only on the training data** and then applied to the test data.

This is important because information from the test set should not be used during preprocessing of the training data.

---

# 🔥 PyTorch Dataset

A custom PyTorch `Dataset` is created:

```python
class RealEstateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

This allows the processed data to work naturally with PyTorch's `DataLoader`.

---

# 📦 DataLoaders

The training and testing datasets are converted into DataLoaders:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
```

### Configuration

| Parameter | Training | Testing |
|---|---:|---:|
| Batch Size | 64 | 64 |
| Shuffle | Yes | No |

Training data is shuffled to improve training behavior, while test data remains in a fixed order.

---

# ⚙️ Loss Function

The model uses **Mean Squared Error (MSE)** as its loss function:

```python
criterion = nn.MSELoss()
```

The model is trained against the **log-transformed target values**.

MSE penalizes larger prediction errors more strongly than smaller errors.

---

# 🚀 Optimizer

The optimizer is **Adam**:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

### Configuration

```text
Optimizer: Adam
Learning Rate: 0.001
Weight Decay: 0.0001
```

The weight decay provides L2-style regularization to help reduce overfitting.

---

# 📉 Learning Rate Scheduler

The project uses `ReduceLROnPlateau`:

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)
```

The scheduler monitors the test loss.

If the loss stops improving for several epochs, the learning rate is reduced by half.

```text
New Learning Rate = Current Learning Rate × 0.5
```

This allows the model to make smaller updates when training reaches a plateau.

---

# 🏋️ Model Training

The model is trained for:

```python
epochs = 50
```

During each epoch:

1. The model is placed in training mode.
2. A batch of data is loaded.
3. Predictions are generated.
4. MSE loss is calculated.
5. Gradients are calculated using backpropagation.
6. The optimizer updates the model parameters.
7. The model is evaluated on the test set.
8. The learning-rate scheduler is updated.

The core training process is:

```python
optimizer.zero_grad()

outputs = model(inputs)

loss = criterion(outputs, targets)

loss.backward()

optimizer.step()
```

The notebook also detects whether a CUDA-compatible GPU is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

In the recorded training run, the model used:

```text
Using device: cuda
```

---

# 📊 Training Results

The recorded training results were:

| Epoch | Train Loss | Test Loss |
|---:|---:|---:|
| 1 | 0.1548 | 0.0372 |
| 10 | 0.0320 | 0.0320 |
| 20 | 0.0284 | 0.0288 |
| 30 | 0.0270 | 0.0277 |
| 40 | 0.0266 | 0.0275 |
| 50 | 0.0262 | 0.0270 |

The loss decreases substantially during training, indicating that the neural network is learning the relationship between the input features and the target.

---

# 📈 Model Evaluation

After training, predictions are converted back from the logarithmic scale:

```python
predictions_original_scale = np.expm1(log_predictions)
y_test_original_scale = np.expm1(y_test)
```

The project evaluates the model using three common regression metrics.

## Mean Absolute Error — MAE

```python
mae = mean_absolute_error(
    y_test_original_scale,
    predictions_original_scale
)
```

Recorded result:

```text
MAE = $37,365.02
```

This means that, on average, the model's predictions differed from the actual values by approximately **$37.4K**.

---

## Root Mean Squared Error — RMSE

```python
rmse = np.sqrt(
    mean_squared_error(
        y_test_original_scale,
        predictions_original_scale
    )
)
```

Recorded result:

```text
RMSE = $55,843.03
```

RMSE gives more weight to larger errors.

---

## R² Score

```python
r2 = r2_score(
    y_test_original_scale,
    predictions_original_scale
)
```

Recorded result:

```text
R² = 0.7620
```

The model therefore explains approximately **76.2% of the variance** in the target values on the evaluated test set.

### Final Results

| Metric | Result |
|---|---:|
| MAE | **$37,365.02** |
| RMSE | **$55,843.03** |
| R² Score | **0.7620** |

> These metrics represent the recorded results from the notebook's specific training run.

---

# 🏠 Example Predictions

The notebook also selects five random samples from the test set and compares their actual and predicted house prices.

Example results from the recorded run:

| Real Price | Predicted Price | Absolute Error |
|---:|---:|---:|
| $234,600.00 | $155,133.42 | $79,466.58 |
| $143,400.00 | $151,463.95 | $8,063.95 |
| $62,100.00 | $49,017.36 | $13,082.64 |
| $500,001.00 | $356,181.62 | $143,819.37 |
| $204,600.00 | $212,719.58 | $8,119.58 |

These examples demonstrate that prediction accuracy varies between individual houses.

---

# 🛠️ Technologies Used

The project was developed using:

- **Python**
- **PyTorch**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Matplotlib**

### Main Libraries

```text
PyTorch       → Deep learning model
Scikit-learn  → Dataset, preprocessing and evaluation
NumPy         → Numerical operations
Pandas        → Data manipulation
Matplotlib    → Visualization
```

---

# 📁 Project Structure

A recommended project structure for turning the notebook into a standalone project would be:

```text
California-Housing-Predictor/
│
├── California_Housing_Predictor_Price_Model.ipynb
├── README.md
├── requirements.txt
│
├── models/
│   └── housing_model.pth
│
└── src/
    ├── preprocessing.py
    ├── model.py
    ├── train.py
    └── predict.py
```

The current implementation is provided as a Jupyter Notebook. The structure above is a recommended organization if the project is later converted into a production-style Python project.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd California-Housing-Predictor
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

Create a `requirements.txt` file containing:

```text
torch
scikit-learn
numpy
pandas
matplotlib
jupyter
```

Then install:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run the Project

The easiest way to run the current implementation is through Jupyter Notebook.

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
California_Housing_Predictor_Price_Model.ipynb
```

Run the cells from top to bottom.

The notebook will:

```text
Load Dataset
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Feature Standardization
     ↓
Create PyTorch Dataset
     ↓
Create DataLoaders
     ↓
Build MLP
     ↓
Configure Adam + MSE
     ↓
Train for 50 Epochs
     ↓
Evaluate Model
     ↓
Calculate MAE / RMSE / R²
     ↓
Generate Predictions
```

---

# 🔮 How Prediction Works

Once the model has been trained, a new housing sample must go through the **same preprocessing pipeline** used during training.

The general prediction pipeline is:

```text
Raw Housing Data
       ↓
Feature Engineering
       ↓
StandardScaler
       ↓
PyTorch Tensor
       ↓
MLP Model
       ↓
Log-Scale Prediction
       ↓
exp(prediction) - 1
       ↓
Original House Value
```

The important point is that new data must be transformed using the **same fitted scaler**:

```python
X_new_scaled = scaler_X.transform(X_new)
```

It should not be fitted again on the new sample.

---

# 🧪 Example Prediction Workflow

After training, the model can be used in evaluation mode:

```python
model.eval()

with torch.no_grad():
    prediction = model(input_tensor)
```

Because the model predicts the logarithmically transformed target, the prediction must be converted back:

```python
prediction_original = np.expm1(prediction)
```

For the California Housing dataset, the target values represent units of **$100,000**, so the notebook converts them to dollar values using:

```python
price_dollars = prediction_original * 100000
```

---

# ⚠️ Important Considerations

### 1. Save the Scaler

If the model is deployed later, the fitted `StandardScaler` must also be saved.

The model alone is not enough because incoming data needs to undergo the same feature scaling.

### 2. Save the Model Weights

The trained PyTorch model can be saved using:

```python
torch.save(model.state_dict(), "housing_model.pth")
```

Then loaded later with:

```python
model.load_state_dict(
    torch.load("housing_model.pth")
)
```

### 3. Keep Feature Order Consistent

The input features must be provided in exactly the same order used during training.

### 4. Apply the Same Feature Engineering

The three engineered features must also be calculated for new data:

```text
RoomsPerHousehold
BedroomsPerRoom
PopulationPerHousehold
```

### 5. Apply the Target Inverse Transformation

The model predicts the transformed target, so predictions need:

```python
np.expm1(...)
```

before converting them to the original scale.

---

# 🚀 Possible Future Improvements

The current model provides a solid deep-learning regression baseline, but it could be extended in several ways:

### Model Improvements

- Add Batch Normalization
- Add Dropout
- Experiment with deeper architectures
- Experiment with different activation functions
- Perform hyperparameter optimization
- Compare different optimizers

### Training Improvements

- Introduce a dedicated validation set
- Implement early stopping
- Save the best model checkpoint
- Track additional training metrics
- Perform cross-validation

### Feature Engineering

Additional domain-specific features could potentially improve predictive performance.

### Deployment

The trained model could eventually be exposed through an API using technologies such as:

```text
FastAPI
      ↓
PyTorch Model
      ↓
Housing Prediction
```

A frontend application could then send housing information to the API and receive a predicted price.

---

# 🎯 Learning Objectives

This project demonstrates several important deep-learning concepts:

- Regression with neural networks
- PyTorch model construction
- Custom PyTorch datasets
- DataLoader usage
- Feature scaling
- Feature engineering
- Log transformations
- Backpropagation
- Adam optimization
- Learning-rate scheduling
- GPU acceleration
- Regression evaluation
- Model inference

It is therefore a practical example of how a tabular regression problem can be solved using a neural network rather than a traditional machine-learning algorithm.

---

# 📊 Results Summary

The final recorded model achieved:

```text
MAE  : $37,365.02
RMSE : $55,843.03
R²   : 0.7620
```

The model was trained for **50 epochs** using:

```text
Architecture : MLP
Optimizer    : Adam
Learning Rate: 0.001
Weight Decay : 0.0001
Loss         : MSE
Activation   : SiLU
Batch Size   : 64
```

The model used GPU acceleration in the recorded run:

```text
Device: CUDA
```

---

# 📜 License

This project is intended for educational and experimental purposes.

If you use or modify this project, please provide appropriate attribution to the original project and dataset source.
