# 🥘 FoodSight AI: The Training Journey Retrospective
**Project:** 100-Class Indian Food Classification  
**Duration:** 4 Days of Iterative Optimization  
**Final Result:** **85.21% Validation Accuracy**

---

## 🏁 Phase 1: Local Hardware & The Windows Barrier
**Objective:** Utilize the local **NVIDIA RTX 5060** for maximum speed.

*   **Roadblock:** Discovered that TensorFlow has officially dropped native GPU support for Windows after version 2.10.
*   **Challenge:** Attempting to force local training resulted in CPU fallback, with an estimated training time of **>20 hours**.
*   **Outcome:** Decision made to shift to Cloud infrastructure (Kaggle/Colab) to meet the Review 4 deadline rather than spending 2 days configuring WSL2.

---

## 🌩️ Phase 2: The Kaggle Struggle (OOM & Data Corruption)
**Objective:** Use Kaggle's Dual T4 GPUs (32GB combined VRAM).

*   **The OOM Wall:** Despite having 32GB of VRAM, the 100-class dataset (38,000+ images) caused repeated **Out Of Memory** crashes. 
*   **The Silent Killer:** We discovered that ~1% of the dataset contained **corrupted/0-byte images**. These would pass the initial loader but crash the training loop mid-way after 30 minutes of training, wasting hours of progress.
*   **The "Small Model" Trap:** Initial attempts with smaller models (EfficientNetB0) showed poor convergence, flatlining at ~40% accuracy.
*   **Outcome:** Kaggle's environment proved difficult to stabilize for a custom training loop with gradient accumulation.

---

## 🛠️ Phase 3: Transition to Google Colab (The "V1-V5" Era)
**Objective:** Stabilize training on a single T4 GPU using customized logic.

*   **Persistence of Errors:** Early versions on Colab still faced **OOM crashes** at 15GB VRAM.
*   **The `ValueError` Roadblock:** We encountered a highly specific TensorFlow error: `Creating variables on a non-first call to a decorated function`. This was caused by the Adam optimizer lazily creating momentum variables inside our custom loop.
*   **Learning Stagnation:** Using a standard `model.fit()` with a batch size of 16 was "too noisy." The model was jumping around and not improving ("The model wasn't improving").

---

## 🚀 Phase 4: The Build of the "Robust V6" Script
**Objective:** Engineer a final, bulletproof solution.

I implemented three critical innovations to break the deadlock:
1.  **Deep Data Scan:** A pre-training step that decodes every image to find and flag corrupt files *before* they can crash the loop.
2.  **Gradient Accumulation:** By accumulating gradients over 4 steps, we simulated a **Batch Size of 56** while only using the memory of 14. This stabilized the learning process.
3.  **Optimizer Pre-Initialization:** We forced the optimizer to create its variables *before* the first epoch, finally resolving the `ValueError` crash.

---

## 🏆 The Result: Success & High Performance
**Final Stage 1 (Transfer Learning):** 74.00%  
**Final Stage 2 (Fine-Tuning):** **85.21%**

*   **Achievement:** On a 100-class problem, 85.2% accuracy is a "Gold Standard" result for mobile architectures (MobileNetV3Large).
*   **Efficiency:** The final training script is now stable enough to run unattended, with automatic saving of the "Best Model" to Google Drive.

---

### **Lessons Learned for Review 5**
1.  **Infrastructure First:** WSL2 setup should be prioritized *after* the review to utilize the RTX 5060.
2.  **Data Integrity:** Always run a "Deep Scan" on new datasets before training.
3.  **Custom Loops over `model.fit`:** When working with large datasets on limited memory, custom loops with gradient accumulation are superior for stability and accuracy.
