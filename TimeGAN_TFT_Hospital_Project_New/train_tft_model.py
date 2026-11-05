

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
import torch

# ------------------------------------
# Load Dataset
# ------------------------------------
df = pd.read_csv("synthetic_department_financial_dataset.csv")

df["date"] = pd.to_datetime(df["date"])
df["time_idx"] = (
    (df["date"].dt.year - df["date"].dt.year.min()) * 12
    + df["date"].dt.month
)

dept_map = {d: i for i, d in enumerate(df["department"].unique())}
df["group_id"] = df["department"].map(dept_map)

# ------------------------------------
# Feature Scaling
# ------------------------------------
continuous_cols = [
    "direct_cost_10k_cny",
    "indirect_allocation_10k_cny",
    "revenue_10k_cny",
    "outpatient_or_exam_volume",
]

scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# ------------------------------------
# Train / Validation Split
# ------------------------------------
max_prediction_length = 6
max_encoder_length = 12

training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="total_cost_10k_cny",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["department_type"],
    time_varying_known_reals=["time_idx", "month_sin", "month_cos"],
    time_varying_unknown_reals=[
        "direct_cost_10k_cny",
        "indirect_allocation_10k_cny",
        "revenue_10k_cny",
        "outpatient_or_exam_volume",
        "total_cost_10k_cny"
    ]
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    df,
    predict=True,
    stop_randomization=True
)

batch_size = 64

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)

# ------------------------------------
# TFT Model
# ------------------------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    loss=QuantileLoss(),
    output_size=3,
    reduce_on_plateau_patience=4,
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=10,
    verbose=False,
    mode="min"
)

trainer = Trainer(
    max_epochs=30,
    accelerator="auto",
    callbacks=[early_stop_callback],
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# ------------------------------------
# Forecasting
# ------------------------------------
predictions = tft.predict(val_dataloader)
print("Prediction tensor shape:", predictions.shape)

print("Training complete.")
