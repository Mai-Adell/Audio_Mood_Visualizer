import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, glob
import librosa
import librosa.display
#import IPython
#from IPython.display import Audio
#from IPython.display import Image
import pickle
import warnings; warnings.filterwarnings('ignore') #matplot lib complains about librosa
# !pip install -U librosa
from pydub import silence, AudioSegment
import IPython.display as ipd
import  cairosvg
#print(cairosvg.__version__)
import  python_avatars  as  pa
#import random as random
#from enum import Enum
#from matplotlib import image
#import time
#import cv2
from moviepy.editor import AudioFileClip,VideoFileClip,concatenate_videoclips
import imageio.v3 as imageio
from concurrent.futures import ThreadPoolExecutor


#from google.colab import drive
#drive.mount("/content/drive")

def make_test(model,criterion):
    def validate_t(X):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)
        return predictions
    return validate_t

def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self,num_emotions):
        super().__init__()

        ################ TRANSFORMER BLOCK #############################
        # maxpool the input feature map/tensor to the transformer
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])

        # define single transformer encoder layer
        # self-attention + feedforward network from "Attention is All You Need" paper
        # 4 multi-head self-attention layers each with 40-->512--->40 feedforward network
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40, # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            nhead=4, # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=512, # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dropout=0.4,
            activation='relu' # ReLU: avoid saturation/tame gradient/reduce compute time
        )

        # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
        # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock1 = nn.Sequential(

            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock2 = nn.Sequential(

            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        ################# FINAL LINEAR BLOCK ####################
        # Linear softmax layer to take final concatenated embedding tensor
        #    from parallel 2D convolutional and transformer blocks, output 8 logits
        # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array
        # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
        # 512*2+40 == 1064 input features --> 8 output emotions
        self.fc1_linear = nn.Linear(512*2+40,num_emotions)

        ### Softmax layer for the 8 output logits from final FC linear layer
        self.softmax_out = nn.Softmax(dim=1) # dim==1 is the freq embedding

    # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self,x):

        ############ 1st parallel Conv2D block: 4 Convolutional layers ############################
        # create final feature embedding from 1st convolutional layer
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding1 = self.conv2Dblock1(x) # x == N/batch * channel * freq * time

        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        # create final feature embedding from 2nd convolutional layer
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding2 = self.conv2Dblock2(x) # x == N/batch * channel * freq * time

        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)


        ########## 4-encoder-layer Transformer block w/ 40-->512-->40 feedfwd network ##############
        # maxpool input feature map: 1*40*282 w/ 1*4 kernel --> 1*40*70
        x_maxpool = self.transformer_maxpool(x)

        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool,1)

        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        x = x_maxpool_reduced.permute(2,0,1)

        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(x)

        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 2x40 (MFCC embedding*time) feature map, take mean of columns i.e. take time average
        transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x70 --> 40

        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        # concatenate embedding tensors output by parallel 2*conv and 1*transformer blocks
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2,transformer_embedding], dim=1)

        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)

        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)

        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax

def saved_model():
  load_folder = 'content/model/'
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  #pick the epoch to load 256 64% -
  #epoch = '415'
  model_name = 'parallel_all_you_wantFINAL-415.pkl'
  load_path = os.path.join(load_folder, model_name)

  ## instantiate empty model and populate with params from binary
  model = parallel_all_you_want(num_emotions=8).to(device)
  optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)
  #print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )
  load_checkpoint(optimizer, model, load_path)
  return model

def load_data(test_Files):
  # raw waveforms to augment later
    waveforms = []
    for file in glob.glob(test_Files):
      waveform = get_waveforms(file)
      waveforms.append(waveform)

    return waveforms

def feature_mfcc(
    waveform,
    sample_rate,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    #hop=256, # increases # of time steps; was not helpful
    mels=128
    ):

    # Compute the MFCCs for all STFT frames
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=fft,
        win_length=winlen,
        window=window,
        #hop_length=hop,
        n_mels=mels,
        fmax=sample_rate/2
        )

    return mfc_coefficients

def get_waveforms(file):
    sample_rate = 48000

    # load an individual sample audio file
    # read the full 3 seconds of the file, cut off the first 0.5s of silence; native sample rate = 48k
    # don't need to store the sample rate that librosa.load returns
    waveform, _ = librosa.load(file, duration=3, sr=sample_rate)

    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate*3,)))
    waveform_homo[:len(waveform)] = waveform

    # return a single file's waveform
    return waveform_homo

def get_features(waveforms, features, sample_rate):

    # initialize counter to track progress
    file_count = 0

    # process each waveform individually to get its MFCCs
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        file_count += 1
        # print progress
       # print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')

    # return all features from list of waveforms
    return features

def final_test(file_test):

  waveforms= load_data(file_test)

  sample_rate = 48000

  # mfcc features
  features=[]
  features_t = get_features(waveforms,features, sample_rate)

  sample_test = np.expand_dims(features,1)

  #print(sample_test.shape)
  N,C,H,W = sample_test.shape

  # load scaler


  with open('content/model/scaler.pkl','rb') as f:
      scaler = pickle.load(f)
  # should save scaler and then load it here to use
  sample_test= np.reshape(sample_test, (N,-1))
  sample_test= scaler.transform(sample_test)
  sample_test = np.reshape(sample_test, (N,C,H,W))

  def criterion(predictions, targets):
      return nn.CrossEntropyLoss()(input=predictions, target=targets)
  model = saved_model()
  validate_t = make_test(model,criterion)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  sample_test= torch.tensor(sample_test,device=device).float()

  predicted_emotion = validate_t(sample_test)
  # print(predicted_emotion)

  emotions_dict ={
      '0':'surprised',
      '1':'neutral',
      '2':'calm',
      '3':'happy',
      '4':'sad',
      '5':'angry',
      '6':'fearful',
      '7':'disgust'
  }
  emotions=[]
  for i in range(len(predicted_emotion)):
    e = predicted_emotion[i].item()
    # this the emotion detected from the audio
    emotion=emotions_dict[str(e)]
    emotions.append(emotion)
   # print(" predicted emotion for segment ",i,": ",emotions[i])
  # # here we will delete this code because the gender will enter from the user in register
  # if (int((file_test.split("-")[6]).split(".")[0]))%2==0:
  #     gender = 'girl'
  # else:
  #     gender = 'boy'
  return emotions

import subprocess

def audio_read(audio_path):
  #!cp $audio_path .
 # print(audio_path)
  subprocess.run(['copy', audio_path, '.'], shell=True)
  original_audio = os.path.basename(audio_path)
  ipd.display(ipd.Audio(original_audio))
 # print(original_audio)
  return original_audio

def read_audio(audio_path):
   # print(audio_path)
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    return audio

def split_on_silence_with_pydub(path_audio,skip_idx=0, out_ext="wav",
        silence_thresh=-51, silence_chunk_len=300, keep_silence=300):
    
    subprocess.run(['copy', path_audio, '.'], shell=True)
    #audio_path=audio_read(path_audio)
    filename = os.path.basename(path_audio).split('.', 1)[0]
    audio = read_audio(path_audio)

    not_silence_ranges = silence.detect_nonsilent(
        audio, min_silence_len=silence_chunk_len,
        silence_thresh=silence_thresh)
    # Save audio files
    audio_paths = []

    for idx, (start_idx, end_idx) in enumerate(not_silence_ranges[skip_idx:]):
        start_idx = max(0, start_idx - keep_silence)
        end_idx += keep_silence
        target_audio_path = "content/segmentation/{}/{}.{:04d}.{}".format(os.path.dirname(path_audio), filename, idx, out_ext)
        segment = audio[start_idx:end_idx]
        segment.export(target_audio_path, out_ext)
        audio_paths.append(target_audio_path)


    return audio_paths,not_silence_ranges

def path_emotions(audio_path,gender):
  audio_list,intervals = split_on_silence_with_pydub(audio_path)
  path='content/segmentation/data/*.wav'
  emotions=final_test(path)
  #print(emotions)
  return emotions

# just to test data found after segmentation
# num=3
# path="/content/drive/MyDrive/avatars_audio/segmentation/sample_audio2.{:04d}.wav".format(num)
# ipd.display(ipd.Audio(path))

def avatar(EB,E,M,TYPE,personality,specific_avatar):
  if personality == "boy":
    #clothing_color, accessory,facial_hair,clothing,top,hair_color
    Arr=['BLUE_01','NONE','MOUSTACHE_MAGNUM','SHIRT_WICK','SHORT_WAVED','BROWN_DARK',
         'GRAY_02','PRESCRIPTION_2','NONE','COLLAR_SWEATER','SHORT_WAVED','BROWN_DARK',
         'BLUE_01','NONE','NONE','HOODIE','QUIFF','BROWN']
  else:
    Arr=['GRAY_02','PRESCRIPTION_2','NONE','SHIRT_SCOOP_NECK','HIJAB','BROWN_DARK',
         'GRAY_02','NONE','NONE','HOODIE','STRAIGHT_1','BROWN_DARK',
         'BLUE_02','PRESCRIPTION_2','NONE','HOODIE','BOB','BROWN_DARK']

  if(specific_avatar==3):
      A=Arr[0]
      B=Arr[1]
      C=Arr[2]
      D=Arr[3]
      F=Arr[4]
      G=Arr[5]
  elif(specific_avatar==2):
      A=Arr[6]
      B=Arr[7]
      C=Arr[8]
      D=Arr[9]
      F=Arr[10]
      G=Arr[11]
  elif(specific_avatar==1):
      A=Arr[12]
      B=Arr[13]
      C=Arr[14]
      D=Arr[15]
      F=Arr[16]
      G=Arr[17]

  if specific_avatar==3 and personality=="girl":
      T=eval('pa.HatType.%s'%F)
  else:
      T=eval('pa.HairType.%s'%F)


  my_avatar = pa.Avatar(
      style=pa.AvatarStyle.CIRCLE,
      background_color=pa.BackgroundColor.WHITE,
      top=T,
      eyebrows=eval('pa.EyebrowType.%s'%EB),
      eyes=eval('pa.EyeType.%s'%E),
      nose=pa.NoseType.DEFAULT,
      mouth=eval('pa.MouthType.%s'%M),
      facial_hair=eval('pa.FacialHairType.%s'%C),
      skin_color="#FFDAB9",
      hair_color=eval('pa.HairColor.%s'%G),
      accessory=eval('pa.AccessoryType.%s'%B),
      clothing=eval('pa.ClothingType.%s'%D),
      clothing_color=eval('pa.ClothingColor.%s'%A)
  )

  # Save to a file
  my_avatar.render("content/avatars/"+TYPE+"_"+personality+".svg")
  example_url ='content/avatars/'+TYPE+'_'+personality+'.svg'
  path='content/avatars/'+TYPE+'_'+personality+'.png'
  s = cairosvg.svg2png(url=example_url,write_to=path)

## disgust
def disgust(type,specific_avatar):
  avatar('DEFAULT_NATURAL','SIDE','CONCERNED',"disgust",type,specific_avatar)
  avatar('DEFAULT_NATURAL','SIDE','SERIOUS',"natural_d",type,specific_avatar)

## surbrise
def surprised(type,specific_avatar):
  avatar('DEFAULT_NATURAL','SURPRISED','DISBELIEF',"surprised",type,specific_avatar)
  avatar('DEFAULT_NATURAL','SURPRISED','SERIOUS',"natural_sur",type,specific_avatar)

## fearful
def fearful(type,specific_avatar):
  avatar('DEFAULT_NATURAL','SQUINT','SCREAM_OPEN',"fearful",type,specific_avatar)
  avatar('DEFAULT_NATURAL','SQUINT','SERIOUS',"natural_f",type,specific_avatar)

## angry
def angry(type,specific_avatar):
  avatar('ANGRY','SQUINT','GRIMACE',"angry",type,specific_avatar)
  avatar('ANGRY','SQUINT','SERIOUS',"natural_a",type,specific_avatar)

## calm
def calm(type,specific_avatar):
  avatar('DEFAULT_NATURAL','CLOSED','TWINKLE',"calm",type,specific_avatar)
  avatar('DEFAULT_NATURAL','CLOSED','SERIOUS',"natural_c",type,specific_avatar)

## happy
def happy(type,specific_avatar):
  avatar('DEFAULT_NATURAL','HAPPY','SMILE',"happy",type,specific_avatar)
  avatar('DEFAULT_NATURAL','HAPPY','SERIOUS',"natural_h",type,specific_avatar)

## sad
def sad(type,specific_avatar):
  avatar('SAD_CONCERNED_NATURAL','CRY','SAD',"sad",type,specific_avatar)
  avatar('SAD_CONCERNED_NATURAL','CRY','SERIOUS',"natural_s",type,specific_avatar)

###UPDATED
## neutral
def neutral(type,specific_avatar):
  avatar('DEFAULT_NATURAL','DEFAULT','TWINKLE',"neutral",type,specific_avatar)
  avatar('DEFAULT_NATURAL','DEFAULT','SERIOUS',"natural_n",type,specific_avatar)

### Updated
def choice(emotion,type,specific_avatar):
  N=" "
  if emotion=="disgust":
        disgust(type,specific_avatar)
        N="d"
  elif emotion=="surprised":
       surprised(type,specific_avatar)
       N="sur"
  elif emotion=="fearful":
        fearful(type,specific_avatar)
        N="f"
  elif emotion=="angry":
       angry(type,specific_avatar)
       N="a"
  elif emotion=="calm":
      calm(type,specific_avatar)
      N="c"
  elif emotion=="happy":
       happy(type,specific_avatar)
       N="h"
  elif emotion=="sad":
      sad(type,specific_avatar)
      N="s"
  elif emotion=="neutral":
      neutral(type,specific_avatar)
      N="n"
  return N

def show(emotion,type,specific_avatar):
  N=choice(emotion,type,specific_avatar)
  path1="content/avatars/"+emotion+"_"+type+".png"
  path2="content/avatars/natural_"+N+"_"+type+".png"
  return path1,path2

# used for test all types of people with their specific_avatars
#show("disgust",'girl',2)

def show_avatar(file_test,emotion,gender,saved_file,specific_avatar):
  path1,path2=show(emotion,gender,specific_avatar)
  images = []
  # here will take the input audio
  #print(file_test)
  audio_clip = AudioFileClip(file_test)
  a=audio_clip.duration
  #print(a)
  # here take paths of the emojies
  for filename in range(int(a*2)):
      images.append(imageio.imread(path1))
      images.append(imageio.imread(path2))
  # Create the output video file
  # here will take the path to save on it the video
  imageio.imwrite(saved_file, images, fps=5)
  video_clip = VideoFileClip(saved_file)
  final_clip = video_clip.set_audio(audio_clip)
  # Export the final video with audio
  final_clip.write_videofile(saved_file)

def final_video(gender,audio_path,specific_avatar):
  
  emotions=path_emotions(audio_path,gender)
  folder='content/segmentation'
  all_saved_files=[]
  for i in range(len(emotions)):
    num_audio='{:04d}'.format(i)
    file_name=os.path.splitext(os.path.basename(audio_path))[0]
    model_name = file_name+f'.{num_audio}.wav'
    saved_file="content/small_audios/output{:04d}.mp4".format(i)
    all_saved_files.append(saved_file)
    # load_path = os.path.join(folder, model_name)
    load_path=f'content/segmentation/data/{file_name}.{num_audio}.wav'
    show_avatar(load_path,emotions[i],gender,saved_file,specific_avatar)
  video_clips=[VideoFileClip(path)for path in all_saved_files]
  conactenated_clips=concatenate_videoclips(video_clips,method="compose")
  output_path= 'final_output.mp4' #'home_test/app/src/main/res/raw/final_output.mp4'
  conactenated_clips.write_videofile(output_path)
  return output_path

#### using parallel make it more faster #####

# def final_video(gender, audio_path, specific_avatar):
#     emotions = path_emotions(audio_path, gender)
#     folder = 'content/segmentation'
#     all_saved_files = []

#     def process_emotion(i):
#         try:
#             num_audio = '{:04d}'.format(i)
#             file_name = os.path.splitext(os.path.basename(audio_path))[0]
#             model_name = file_name + f'.{num_audio}.wav'
#             saved_file = "content/small_audios/output{:04d}.mp4".format(i)
#             load_path = f'content/segmentation/data/{file_name}.{num_audio}.wav'#os.path.join(folder, model_name)
#             show_avatar(load_path, emotions[i], gender, saved_file, specific_avatar)
#             return saved_file
#         except Exception as e:
#             print("-------------------------------------------")
#             print(e)
#             print(f"Error processing emotion {i}: {e}")
#             print("-------------------------------------------")
#             return None

#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_emotion, i) for i in range(len(emotions))]
#         all_saved_files = [future.result() for future in futures if future.result() is not None]


#     video_clips = [VideoFileClip(path) for path in all_saved_files]
#     concatenated_clips = concatenate_videoclips(video_clips, method="compose")
#     output_path = 'final_output.mp4'
#     concatenated_clips.write_videofile(output_path)
#     return output_path


