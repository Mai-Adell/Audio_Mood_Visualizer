package com.iti.home_gp_test;

import static android.content.ContentValues.TAG;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import androidx.fragment.app.Fragment;
import androidx.annotation.NonNull;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class home_record_fragment extends Fragment {

    private MediaRecorder mediaRecorder;
    private String audioFilePath;
    private boolean isRecording = false;
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private MediaPlayer mediaPlayer;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view = inflater.inflate(R.layout.fragment_home_record_fragment, container, false);

        ImageButton btnRecord = view.findViewById(R.id.btn_record);
        Button btnPlay = view.findViewById(R.id.btnPlay);

        Context context = requireContext();

        audioFilePath = context.getExternalCacheDir().getAbsolutePath() + "/audio.wav";

        // Request permission to record audio
        // Request permission to record audio
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO_PERMISSION);
            //return;
        }

        // Set up MediaRecorder
        mediaRecorder = new MediaRecorder();
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
        mediaRecorder.setOutputFile(audioFilePath);

        // Set up MediaPlayer
        mediaPlayer = new MediaPlayer();

        btnRecord.setOnClickListener(v -> {
            if (!isRecording) {
                try {
                    // Start recording
                    mediaRecorder.prepare();
                    mediaRecorder.start();
                    Log.d("AudioFilePath", "Audio File Path: " + audioFilePath);
                    isRecording = true;
                    Toast.makeText(context,"recordining " ,Toast.LENGTH_SHORT).show();

                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else {
                Log.d("AudioFilePath", "Audio File Path: " + audioFilePath);

                // Stop recording
                mediaRecorder.stop();
                mediaRecorder.release();
                isRecording = false;
                Toast.makeText(context,"record stopped " ,Toast.LENGTH_SHORT).show();


                // Enable play button
                btnPlay.setEnabled(true);

                // Upload the audio file
                uploadAudioFile(audioFilePath);
            }
        });

        btnPlay.setOnClickListener(v -> {
            try {
                if (mediaPlayer.isPlaying()) {
                    mediaPlayer.stop();
                }
                mediaPlayer.reset();
                mediaPlayer.setDataSource(audioFilePath);
                mediaPlayer.prepare();
                mediaPlayer.start();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });


        return view;
    }

    private void uploadAudioFile(String filePath) {
        OkHttpClient client = new OkHttpClient();

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", "audio.wav",
                        RequestBody.create(MediaType.parse("audio.wav"), new File(filePath)))
                .build();

        Request request = new Request.Builder()
                .url("http://192.168.1.3:5000/upload")
                .post(requestBody)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                e.printStackTrace();
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    throw new IOException("Unexpected code " + response);
                }

                // Handle successful response
                String responseData = response.body().string();
                Log.d(TAG, "Response: " + responseData);
            }
        });
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (isRecording) {
            mediaRecorder.stop();
            mediaRecorder.release();
        }
        if (mediaPlayer != null) {
            mediaPlayer.release();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, initialize MediaRecorder
                mediaRecorder = new MediaRecorder();
                mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
                mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
                mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
                mediaRecorder.setOutputFile(audioFilePath);
            } else {
                // Permission denied, show a message or handle it accordingly
            }
        }
    }

}

