# Multimodal Property Price Prediction Using Satellite Imagery

## Overview
This project builds a **multimodal machine learning pipeline** to predict residential property prices by combining:
- **Structured tabular features** (property attributes, location, quality)
- **Satellite imagery features** extracted using a pretrained Convolutional Neural Network (CNN)

The primary objective is to evaluate whether **satellite image representations provide incremental predictive power** beyond traditional tabular real estate features.

---

## Problem Statement
Traditional property valuation models rely heavily on structured data such as square footage, number of bedrooms, and location. However, satellite imagery may encode additional contextual information such as:
- Urban density
- Green cover
- Road connectivity
- Neighborhood structure

This project investigates whether such visual signals improve price prediction accuracy when combined with tabular data.

---

## Data Description

### 1. Tabular Features
Key structured inputs include:
- Bedrooms, bathrooms
- Living area and lot size
- Year built / renovated
- Condition and grade
- Latitude and longitude
- Zipcode-level location features

### 2. Satellite Images
- One satellite image per property
- Images correspond to property coordinates
- Images are not downloaded and directly processed to extract features
- Each image is mapped to a property using a unique ID

---

## Methodology

### Image Feature Extraction
- A **ResNet18** model pretrained on **ImageNet** is used
- The final classification layer is removed
- Each image is converted into a **512-dimensional embedding**
- No fine-tuning is performed (feature extractor only)

```text
Satellite Image
   ↓
ResNet18 (pretrained, classifier removed)
   ↓
512-dim Image Embedding

[ Tabular Features ] + [ Image Embeddings ] → Regression Model → Price
