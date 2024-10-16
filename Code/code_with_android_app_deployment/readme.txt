Android studio:

-  IDE version 2023.2.1
-  sourceCompatibility JavaVersion.VERSION_1_8
-  targetCompatibility JavaVersion.VERSION_1_8
-  minSdk 23
-  targetSdk 33
-  gradle 8.4

plugins {
    id 'com.android.application' version '8.3.0' apply false
    id 'com.android.library' version '8.3.0' apply false
    id 'org.jetbrains.kotlin.android' version '1.9.0' apply false
    id 'org.jetbrains.kotlin.jvm' version '1.9.0'
}


python:

Python 	3.9.7
IPython	7.19.0
Flask 	1.1.2
torch 	2.1.1
matplotlib  3.4.3
numpy 	1.20.3
pandas 	1.3.4
glob2 	0.7
librosa 	0.10.1
pickleshare 0.7.5
pydub 	0.25.1
CairoSVG 2.7.1
python_avatars 1.4.0
moviepy 	1.0.3
imageio 	2.34.0
mkl-random       1.2.2

############################################################################

-  Audio file must be .wav and no longer that 30 sec while using the application as the API give time out            exception while creating the video.

-  If using model_test_script.py there is no limit for lengh.

-  First to run the application you have to have a IDE like spyder to run only the API.py file and leave it runing,          then open the android studio and open e_android_application file and run with your mobile or emulator.

-  While using the app make sure that you make access for network and your IPv4 Address: 192.168.1.6 if not you    have to change it in the code to use it freely.

-  Make sure the internet is stable so it won't make reset execption.

- Give access to read external storage, and write external storage so the application could perform right.

############################################################################

In code file:

-  e_android_application: is the file of the android application.

-  API.py: is the used API. 

-  model_test_script.py: is the model you could call -  function final_video() to run the model seperatelly.

-  content & data files: for the application and model to use.
 






