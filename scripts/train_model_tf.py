import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os
import sys
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

# Add current directory to path to import dataset_loader_tf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_loader_tf import load_dataset

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30  # Increased to 30 as per safe starting point
FINE_TUNE_EPOCHS = 30
LEARNING_RATE = 0.001

def create_model(num_classes):
    print(f"Creating model for {num_classes} classes...")
    
    # Base model: MobileNetV3Large (pre-trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False  # Freeze base model initially

    # Data Augmentation Wrapper
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False) # Important for BatchNormalization
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def get_history_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    return acc, val_acc, loss, val_loss

def plot_history(history, save_dir):
    acc, val_acc, loss, val_loss = get_history_metrics(history)
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plot_path = os.path.join(save_dir, "training_results.png")
    plt.savefig(plot_path)
    print(f"Saved training plot to {plot_path}")
    return acc, val_acc, loss, val_loss

def evaluate_model(model, test_ds, class_names, results_dir):
    print("\n" + "=" * 50)
    print("GENERATING PERFORMANCE REPORTS")
    print("=" * 50)
    
    if not test_ds:
        print("No test dataset available for evaluation.")
        return

    # Get true labels and predictions
    y_true = []
    y_pred = []
    
    print("Evaluating on test set...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(results_dir, "classification_report.csv")
    report_df.to_csv(report_path)
    print(f"Saved classification report to {report_path}")

    # 2. Confusion Matrix (Full matrix for 100 classes)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(24, 24))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, annot=False, cmap='Blues')
    plt.title('Confusion Matrix (All 100 Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # 3. Class-wise F1 Score Bar Chart
    f1_scores = {name: report[name]['f1-score'] for name in class_names if name in report}
    sorted_f1 = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Plot Top 20
    plt.figure(figsize=(12, 10))
    top_20 = sorted_f1[:20]
    plt.barh([x[0] for x in top_20], [x[1] for x in top_20], color='green')
    plt.title('Top 20 Best Performing Classes (F1-Score)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "top_20_classes.png"))
    plt.close()

    # Plot Bottom 20
    plt.figure(figsize=(12, 10))
    bottom_20 = sorted_f1[-20:]
    plt.barh([x[0] for x in bottom_20], [x[1] for x in bottom_20], color='red')
    plt.title('Bottom 20 Classes Needing Improvement (F1-Score)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bottom_20_classes.png"))
    plt.close()

    print("✅ All performance graphs (Confusion Matrix, Top/Bottom Classes) saved to 02_Results_And_Evaluation folder.")

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # FoodSight-AI-App
    data_dir = os.path.join(base_dir, "dataset", "train")
    results_dir = os.path.join(base_dir, "..", "02_Results_And_Evaluation") # External results folder
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"Dataset Directory: {data_dir}")
    print(f"Results Directory: {results_dir}")

    # Load Data (physical folders)
    try:
        train_ds, val_ds, test_ds, class_names = load_dataset(os.path.dirname(data_dir), IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    except Exception as e:
        print(f"Error: {e}")
        return

    num_classes = len(class_names)
    
    # Save class names
    with open(os.path.join(results_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(class_names))

    # Determine Strategy (GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy() # Use all GPUs
        print(f"Running on {len(gpus)} GPU(s) with MirroredStrategy.")
    else:
        strategy = tf.distribute.get_strategy() # Default (CPU)
        print("Running on CPU.")

    with strategy.scope():
        model = create_model(num_classes)
        model.summary()

    # Callbacks
    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_path = os.path.join(results_dir, "best_model.keras")
    checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
    
    log_dir = os.path.join(results_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[stop_early, checkpoint, tensorboard_callback]
    )

    # Save final model
    print(f"Final model (Head) saved to {final_model_path}")

    # Plot results
    acc, val_acc, loss, val_loss = plot_history(history, results_dir)

    # ==============================================================================
    # FINE-TUNING PHASE
    # ==============================================================================
    print("\n" + "=" * 50)
    print("STARTING FINE-TUNING PHASE")
    print("=" * 50)
    
    # Unfreeze the base model
    base_model = model.layers[2] # Layer 0 is Input, 1 is Augmentation, 2 is MobileNetV3Large
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    # MobileNetV3Large has many layers, let's fine-tune the top 50 (or fewer if unstable)
    # fine_tune_at = 100 
    # for layer in base_model.layers[:fine_tune_at]:
    #     layer.trainable = False
    
    print("Number of layers in the base model: ", len(base_model.layers))
    # print(f"Fine-tuning from layer {fine_tune_at} onwards.")

    # Recompile with a lower learning rate
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Low LR is critical
                  metrics=['accuracy'])

    # Fine-tuning epochs
    # FINE_TUNE_EPOCHS defined at top
    total_epochs =  EPOCHS + FINE_TUNE_EPOCHS

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        callbacks=[stop_early, checkpoint, tensorboard_callback]
    )

    # Save Fine-Tuned Model
    fine_tuned_model_path = os.path.join(base_dir, "food_classifier_mobilenet_finetuned.keras")
    model.save(fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")
    
    # Update app to use this model if desired (or user manually renamed)
    # For now, we keep distinct names.

    # Plot Fine-tuning Results
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']
    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([EPOCHS-1,EPOCHS-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([EPOCHS-1,EPOCHS-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    
    finetune_plot_path = os.path.join(results_dir, "training_results_finetuned.png")
    finetune_plot_path = os.path.join(results_dir, "training_results_finetuned.png")
    plt.savefig(finetune_plot_path)
    print(f"Saved fine-tuning plot to {finetune_plot_path}")

    # Final Comprehensive Evaluation
    evaluate_model(model, test_ds, class_names, results_dir)

if __name__ == "__main__":
    main()
