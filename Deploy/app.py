from prediction import Prediction
from DataPreprocessing.recorder import Recorder
import gc
def main():   
    try:
        while True:
            rec = Recorder()
            print("Recording...")
            chunk = rec.record()
            pred = Prediction(chunk)
            print(pred.predict())
            
            del chunk
            del rec
            gc.collect()

    except KeyboardInterrupt:
        print("Recording stopped")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()