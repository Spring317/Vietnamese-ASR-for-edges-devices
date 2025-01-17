import sounddevice as sd 


class Recorder:
    
    """Recording audio from a device for a certain duration (default 4 seconds)"""
    
    def __init__(self, duration=4, sr=44100, channels = 1):
        self.__duration = duration
        self.__sr = sr
        self.__channel = channels
        sd.default.device = "Microphone (High Definition Audio Device), Windows WASAPI"

    def record(self):
        audio =  sd.rec(int(self.__duration * self.__sr), samplerate=self.__sr, channels=self.__channel)
        sd.wait()
        audio = audio.squeeze()
        return audio 
        # sd.wait()
        # wv.write(file_path, audio, self.sr, sampwidth=2)
        # print(audio)
        
# def main():
#     rec = Recorder()
#     print("Recording...")
#     rec.record("test.wav")

# if __name__ == '__main__':
#     main()
