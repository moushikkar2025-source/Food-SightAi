# FoodSight AI - Executive Summary
## Review 4 Submission

**Project**: AI-Based Food Classification for Calorie and Nutrition Estimation  
**Team**: Multidisciplinary (CSE + VLSI)  
**Review Stage**: Review 4 - Advanced Features & Expansion  
**Date**: February 2026

---

## 🎯 Project Objective

To evolve the FoodSight AI system from a simple image classifier into a comprehensive **Nutrition Assistant**. The system now not only identifies *what* food is present but also provides detailed nutritional breakdowns and adjusts for serving sizes, paving the way for real-world diet tracking applications.

---

## 🏆 Key Achievements (Review 4)

### 1. 📈 100-Class Model with MobileNetV3Large
We scaled the model from 20 → 45 → **100 Indian food classes**, covering a comprehensive range of regional cuisines.
- **Data Collection**: Combined automated **web scraping** (Google Images, food blogs) with a strategic merge of the **Khana** dataset (~148,000 images, 80 classes).
- **Dataset Audit**: Used custom scripts (`audit_script.py`, `merge_datasets.py`) to identify gaps, resolve naming conflicts, and rebalance the expanded dataset.
- **Architecture**: **MobileNetV3Large** (pretrained on ImageNet, with built-in preprocessing).
- **Framework**: Migrated to **TensorFlow/Keras** for robust training and deployment.
- **Training**: Custom training loop on Google Colab (T4 GPU) with gradient accumulation (effective batch=56), mixup augmentation, and label smoothing.
- **Fine-Tuning**: Unfroze top 30 base layers with lr=1e-5 for accuracy refinement.
- **Model Size**: 3,092,452 parameters — lightweight enough for real-time inference.

### 2. 🍎 Nutrition Intelligence Integration
We have successfully transformed the classification results into actionable health data.
- **Database**: Comprehensive nutrition database (`nutrition_data.py`) covering **100 Indian dish classes**.
- **Metrics**: Provides granular data for **Calories, Protein, Carbohydrates, Fats, Fiber**, and **Sodium**.
- **Dietary Flags**: Automatically tags dishes as Vegetarian, Vegan, Gluten-Free, or Spicy.
- **Health Indicators**: Automatically calculated badges (High Protein, Heart Healthy, Diabetic Friendly, etc.).
- **Frontend**: Modern "Nutrition Card" UI with color-coded macro-nutrient bars and circular SVG charts.
- **Data Sources**: USDA FoodData Central and Indian Food Composition Tables (IFCT 2017, NIN/ICMR).

### 4. 🥣 Dynamic Portion Size Estimation
Recognizing that "one size fits all" fails in diet tracking, we implemented a dynamic portion scaling system.
- **User Control**: Intuitive **Portion Selector** (Small, Medium, Large, Family) in the results interface.
- **Smart Scaling**: Backend logic (`portion_data.py`) automatically recalculates all nutritional values based on standard multipliers (0.5x to 2.0x).
- **Real-time Updates**: The UI updates instantly without reloading, providing a seamless user experience.

### 5. 🔐 User Accounts & Persistent Tracking
Shifted from a local-only experience to a cloud-ready platform with user identities.
- **Security**: Robust authentication using **bcrypt** password hashing and **Flask-Login** session management.
- **Persistence**: A dedicated `ScanHistory` database tracks every meal a user logs, accessible across browser sessions.
- **Privacy-First**: Implemented a "Clear History" function and zero-photo-retention policy for the persistent scan log to safeguard user data.

### 6. 🔍 Search-Based Nutrition Lookup
Integrated a "Universal Search" bar to supplement the AI camera scanner.
- **Text Queries**: Instant access to nutrition for any of the 100+ dishes via simple text input.
- **Feature Parity**: Search results share the same rich macro-visualization and portion estimation tools as the image-based scanner.
- **Feedback Loop**: Unknown items are flagged for future updates, helping guide future dataset expansion efforts.

### 7. 🛡️ Platform Security & Hardening
Hardened the application for real-world deployment.
- **Rate Limiting**: Throttling on prediction and search endpoints to prevent DoS and scraping.
- **Input Sanitization**: UUID-based renaming and MIME-type validation for all file uploads.
- **Infrastructure**: Complete secrets management using `.env` and secure session cookie configuration.

---

## 🔬 Technical Implementation

### Architecture Updates
*   **Backend (Flask)**: Enhanced API endpoints (`/api/predict`, `/api/nutrition`, `/api/classes`, `/api/dataset/info`) with TensorFlow model integration.
*   **Deep Learning**: **TensorFlow/Keras** with **MobileNetV3Large**. Model loaded via `tf.keras.models.load_model()` for efficient inference.
*   **Frontend (JS/CSS)**: Glassmorphism design with circular macro charts, staggered animations, and history sidebar.
*   **Data Structure**: Modularized code into `dataset_config.py`, `nutrition_data.py`, and `portion_data.py` for maintainability.

### Performance
*   **Inference Speed**: Remains **<1 second** on CPU.
*   **Input**: 224×224 RGB images (raw [0, 255] — model includes internal preprocessing/rescaling).
*   **Output**: Softmax probabilities across 100 classes.
*   **Scalability**: The architecture supports adding new classes by updating configuration and retraining.

---

## 📊 Visual Validation
*   **Training Dashboard**: Documented training progress across epochs.
*   **UI Screenshots**: Captured homepage, results (medium & large portions), demonstrating the full user flow.
*   **Live Demo**: The web application is fully functional with all features integrated.

---

## 🚧 Challenges & Solutions

| Challenge | Solution |
| :--- | :--- |
| **OOM on Colab GPU** | Strict 13GB memory limits, gradient accumulation (batch=14, accum=4), and `gc.collect()` per epoch. |
| **Corrupt Dataset Images** | Deep data scan with `tf.image.decode_image` to detect and skip corrupted files before training. |
| **Keras 3 Compatibility** | Used helper functions for metric resets (`reset_states` vs `reset_state`) and pre-initialized optimizer slots. |
| **Portion Accuracy** | Adopted a multiplier-based heuristic approach for Review 4, with plans for CV-based volumetric estimation. |

---

## 🔮 Future Scope & Conclusion

The FoodSight AI project has successfully met the Review 4 objectives. The system is no longer just a "Food Identifier" but a **"Smart Dietician"** capable of recognizing 100 dishes.

**Next Steps**:
1.  **Hardware**: Proceed with the FPGA implementation plan outlined in the Hardware Acceleration documentation.
2.  **Health Trends**: Implement weekly/monthly nutrition trend visualization.
3.  **Production Readiness**: Containerizing the application for secure cloud deployment.

**Status**: ✅ **100-Class Model Integrated** | ✅ **Security Hardening Complete** | ✅ **User History & Search Live**
