"""TeleRetain AI Stage 5: strict, dataset-specific Pydantic v2 input contracts.

Public dataset columns are the canonical API field names (aliases).
No real Teletalk customer information is used or accepted as a data source.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=False)


class IranianInput(_StrictFeatures):
    call_failure: int = Field(alias="Call Failure", ge=0, strict=True)
    complains: int = Field(alias="Complains", ge=0, strict=True)
    subscription_length: int = Field(alias="Subscription Length", ge=0, strict=True)
    charge_amount: int = Field(alias="Charge Amount", ge=0, strict=True)
    seconds_of_use: int = Field(alias="Seconds of Use", ge=0, strict=True)
    frequency_of_use: int = Field(alias="Frequency of use", ge=0, strict=True)
    frequency_of_sms: int = Field(alias="Frequency of SMS", ge=0, strict=True)
    distinct_called_numbers: int = Field(alias="Distinct Called Numbers", ge=0, strict=True)
    age_group: int = Field(alias="Age Group", ge=0, strict=True)
    tariff_plan: int = Field(alias="Tariff Plan", ge=0, strict=True)
    status: int = Field(alias="Status", ge=0, strict=True)
    age: int = Field(alias="Age", ge=0, strict=True)
    customer_value: float = Field(alias="Customer Value", ge=0, allow_inf_nan=False)


class IBMInput(_StrictFeatures):
    gender: str = Field(alias="gender", min_length=1)
    senior_citizen: Literal["0", "1"] = Field(alias="SeniorCitizen")
    partner: str = Field(alias="Partner", min_length=1)
    dependents: str = Field(alias="Dependents", min_length=1)
    tenure: int = Field(alias="tenure", ge=0, strict=True)
    phone_service: str = Field(alias="PhoneService", min_length=1)
    multiple_lines: str = Field(alias="MultipleLines", min_length=1)
    internet_service: str = Field(alias="InternetService", min_length=1)
    online_security: str = Field(alias="OnlineSecurity", min_length=1)
    online_backup: str = Field(alias="OnlineBackup", min_length=1)
    device_protection: str = Field(alias="DeviceProtection", min_length=1)
    tech_support: str = Field(alias="TechSupport", min_length=1)
    streaming_tv: str = Field(alias="StreamingTV", min_length=1)
    streaming_movies: str = Field(alias="StreamingMovies", min_length=1)
    contract: str = Field(alias="Contract", min_length=1)
    paperless_billing: str = Field(alias="PaperlessBilling", min_length=1)
    payment_method: str = Field(alias="PaymentMethod", min_length=1)
    monthly_charges: float = Field(alias="MonthlyCharges", ge=0, allow_inf_nan=False)
    # Eleven missing values were observed in original IBM data after numeric conversion.
    # Only this field may be null; the train-fitted pipeline will impute it.
    total_charges: float | None = Field(alias="TotalCharges", ge=0, allow_inf_nan=False)

    @field_validator("senior_citizen", mode="before")
    @classmethod
    def normalize_senior_citizen(cls, value):
        # CSV inspection loads this indicator as integer, but training encoded strings.
        if type(value) is int and value in (0, 1):
            return str(value)
        return value


class PredictionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track: Literal["iranian", "ibm"]
    data_source_label: str
    data_source_description: str
    selected_model: str
    churn_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    decision_threshold: float = Field(gt=0, lt=1)
    predicted_class: Literal[0, 1]
    predicted_label: Literal["No churn", "Churn"]
    disclaimer: str


TRACK_SCHEMAS = {"iranian": IranianInput, "ibm": IBMInput}
