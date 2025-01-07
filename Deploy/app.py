from prediction import Prediction

def main():
   
    file_path = "G:/Vietnamese-ASR-for-edges-devices/Deploy/Test_audio/VIVOSSPK01_R003.wav"
    pred = Prediction(file_path)
    print(pred.predict())

if '__name__' == '__main__':
    main()