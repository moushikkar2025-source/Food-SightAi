import tensorflow as tf
import os

def check_gpu():
    print("TensorFlow Version:", tf.__version__)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[+] GPU Available: {len(gpus)} GPU(s) detected.")
        for gpu in gpus:
            print(f"  - {gpu}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"    - Memory growth enabled for {gpu.name}")
            except RuntimeError as e:
                print(f"    - Memory growth error: {e}")
    else:
        print("[!] No GPU detected. Running on CPU.")

    print("\nCUDA Support:", tf.test.is_built_with_cuda())

if __name__ == "__main__":
    check_gpu()
