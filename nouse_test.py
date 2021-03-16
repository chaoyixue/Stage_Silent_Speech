import librosa

sound, sample_rate = librosa.load("../data/20200617_153719_RecFile_1_bruce_ch7"
                                  "/RecFile_1_20200617_153719_Sound_Capture_DShow_5_monoOutput1.wav", 44100)
print(sound.shape)
print(librosa.get_duration(filename="../data/20200617_153719_RecFile_1_bruce_ch7"
                           "/RecFile_1_20200617_153719_Sound_Capture_DShow_5_monoOutput1.wav", sr=44100))