package com.iti.home_gp_test;

import static android.content.ContentValues.TAG;


import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.MediaController;
import android.widget.Toast;
import android.widget.VideoView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class home_upload_fragment extends Fragment {
    private VideoView videoView;
    private Button button_send_post;
    private ALodingDialog aLodingDialog;
    LinearLayout frame_btn_record;
    private static final int REQUEST_PERMISSION_CODE = 123;
    private int pageNumber;
    private String gender;

    public static home_upload_fragment newInstance(String gender,int pageNumber) {
        home_upload_fragment fragment = new home_upload_fragment();
        Bundle args = new Bundle();
        args.putString("gender", gender);
        args.putInt("page_number", pageNumber);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {


        View view = inflater.inflate(R.layout.fragment_home_upload_fragment, container, false);
        Context context = requireContext();


        Bundle args = getArguments();
        if (args != null) {
            pageNumber = args.getInt("page_number");
            gender = args.getString("gender");
            // Do something with the data
        }

        //Toast.makeText(context, "upload_Genderrrrrrrr: " + gender, Toast.LENGTH_SHORT).show();
        //Toast.makeText(context, "upload_page_number: " + pageNumber, Toast.LENGTH_SHORT).show();


        frame_btn_record = view.findViewById(R.id.frame_btn_record);

        if (context.checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_PERMISSION_CODE);
        }
        if (context.checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_PERMISSION_CODE);
        }
        if (context.checkSelfPermission(Manifest.permission.INTERNET) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.INTERNET}, REQUEST_PERMISSION_CODE);
        }



        frame_btn_record.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                call_audio_from_device();
            }
        });

        videoView = view.findViewById(R.id.videoView);
        MediaController mediaController = new MediaController(context);

        button_send_post = view.findViewById(R.id.button);
        aLodingDialog = new ALodingDialog(context);

        button_send_post.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                OkHttpClient client = new OkHttpClient.Builder()
                        .connectTimeout(100, TimeUnit.SECONDS)
                        .readTimeout(100, TimeUnit.SECONDS)
                        .writeTimeout(100, TimeUnit.SECONDS)
                        .build();


                RequestBody requestBody = new MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart("specific_avatar", String.valueOf(pageNumber))
                        .addFormDataPart("gender", gender)
                        .build();

                Request request = new Request.Builder()
                        .url("http://192.168.1.6:5000/retrive")
                        .post(requestBody)
                        .build();
                aLodingDialog.show();

                Handler handler = new Handler();
                Runnable runnable = new Runnable() {
                    @Override
                    public void run() {
                        aLodingDialog.cancel();
                    }
                };

                //handler.postDelayed(runnable,100000);

                client.newCall(request).enqueue(new Callback() {
                    @Override
                    public void onFailure(Call call, IOException e) {
                        e.printStackTrace();
                        // Remove the runnable from the handler queue if the request fails
                        handler.removeCallbacks(runnable);
                        aLodingDialog.cancel();
                        getActivity().runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(requireContext(), "call failed", Toast.LENGTH_SHORT).show();
                            }
                        });

                    }

                    @Override
                    public void onResponse(Call call, Response response) throws IOException {
                        if (!response.isSuccessful()) {
                            handler.removeCallbacks(runnable);
                            aLodingDialog.cancel();
                            getActivity().runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(requireContext(), "call not successful (weird output)", Toast.LENGTH_SHORT).show();
                                }
                            });

                            throw new IOException("Unexpected code " + response);
                        }

                        // Handle successful response
                        String responseData = response.body().string();
                        Log.d(TAG, "Response: " + responseData);

                        handler.removeCallbacks(runnable);
                        aLodingDialog.cancel();

                        downloadVideoWithDelay(100);
                    }

                });
            }
        });

        return view;
    }

    private void call_audio_from_device() {

        Intent intent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Audio.Media.EXTERNAL_CONTENT_URI);//(Intent.ACTION_OPEN_DOCUMENT);
        //intent.addCategory(Intent.CATEGORY_OPENABLE);
        //intent.setType("audio/*");
        //intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(intent, 1);
    }

    @Override
    public void onActivityResult(int requset_code, int result_code, Intent resultData) {

        super.onActivityResult(requset_code, result_code, resultData);

        if (requset_code == 1 && result_code == Activity.RESULT_OK) {
            if (resultData != null) {
                Uri uri = resultData.getData();
                if (uri != null) {
                    String selectedAudioPath = uri.toString();
                    Toast.makeText(requireContext(), selectedAudioPath, Toast.LENGTH_SHORT).show();
                    uploadAudioFile(selectedAudioPath);
                } else {
                    Toast.makeText(requireContext(), "Selected audio URI is null", Toast.LENGTH_SHORT).show();
                }
            } else {
                Toast.makeText(requireContext(), "Intent data is null", Toast.LENGTH_SHORT).show();

            }
        }

    }

    private void uploadAudioFile(String filePath) {
        try {
            InputStream inputStream = requireContext().getContentResolver().openInputStream(Uri.parse(filePath));
            File file = createFileFromInputStream(inputStream);

            OkHttpClient client = new OkHttpClient.Builder()
                    .connectTimeout(100, TimeUnit.SECONDS)
                    .readTimeout(100, TimeUnit.SECONDS)
                    .writeTimeout(100, TimeUnit.SECONDS)
                    .build();

            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("file", "audio1.wav",
                            RequestBody.create(MediaType.parse("audio.wav"), file))
                    .build();

            Request request = new Request.Builder()
                    .url("http://192.168.1.6:5000/upload")
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
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private File createFileFromInputStream(InputStream inputStream) {
        try {
            File file = new File(requireContext().getCacheDir(), "audio_temp.wav");
            OutputStream outputStream = new FileOutputStream(file);
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }
            outputStream.flush();
            outputStream.close();
            inputStream.close();
            return file;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public void downloadVideo() {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    OkHttpClient client = new OkHttpClient.Builder()
                            .connectTimeout(30, TimeUnit.SECONDS)
                            .readTimeout(30, TimeUnit.SECONDS)
                            .writeTimeout(30, TimeUnit.SECONDS)
                            .build();

                    String url = "http://192.168.1.6:5000/video";

                    String directoryPath = String.valueOf(requireActivity().getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS));//"/storage/emulated/0/Download/";
                    String fileName = "final_output.mp4";
                    File destinationFile = new File(directoryPath, fileName);

                    /*if (destinationFile.exists()) {
                        destinationFile.delete();
                    }*/

                    Request request = new Request.Builder()
                            .url(url)
                            .get()
                            .build();

                    try (Response response = client.newCall(request).execute())
                    {
                        if (!response.isSuccessful()) {

                            requireActivity().runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(requireContext(), "call not successful (weird output)", Toast.LENGTH_LONG).show();
                                }
                            });

                            throw new IOException("Unexpected code " + response);
                        }
                        requireActivity().runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(requireContext(), "Video on cloud ...!!!", Toast.LENGTH_LONG).show();
                            }
                        });
                        try (InputStream inputStream = response.body().byteStream();
                             FileOutputStream outputStream = new FileOutputStream(destinationFile))
                        {
                            byte[] buffer = new byte[8192];
                            int bytesRead;
                            while ((bytesRead = inputStream.read(buffer)) != -1) {
                                outputStream.write(buffer, 0, bytesRead);
                            }
                        }catch (Exception e)
                        {
                            requireActivity().runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(requireContext(), "Error Occur While saving the Video...!!!", Toast.LENGTH_LONG).show();
                                }
                            });
                            e.printStackTrace();
                        }
                    }
                    requireActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            playDownloadedVideo();
                        }
                    });

                } catch (IOException e)
                { //on failure

                    requireActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(requireContext(), "call failed", Toast.LENGTH_LONG).show();
                        }
                    });

                    e.printStackTrace();
                }
            }
        });
        thread.start();
    }

    private void playDownloadedVideo() {
        //File videoFile = new File("/storage/emulated/0/Download/final_output.mp4");
        String directoryPath = String.valueOf(requireActivity().getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS));//"/storage/emulated/0/Download/";
        String fileName = "final_output.mp4";
        File videoFile = new File(directoryPath, fileName);
        if (videoFile.exists()) {
            Uri videoUri = Uri.fromFile(videoFile);
            videoView.setVideoURI(videoUri);
            MediaController mediaController = new MediaController(requireContext());
            mediaController.setAnchorView(videoView);
            videoView.setMediaController(mediaController);
            videoView.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
                @Override
                public void onPrepared(MediaPlayer mp) {
                    videoView.pause();
                    //videoView.start();
                    mediaController.show(2000);
                }
            });

            videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                @Override
                public void onCompletion(MediaPlayer mp) {
                    mediaController.show(2000);
                    //mediaController.hide();
                }
            });

            videoView.setOnErrorListener(new MediaPlayer.OnErrorListener() {
                @Override
                public boolean onError(MediaPlayer mp, int what, int extra) {
                    requireActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(requireContext(), "Oops An Error Occur While Playing Video...!!!", Toast.LENGTH_LONG).show();
                        }
                    });
                    return false;
                }
            });



        }else
        {
            requireActivity().runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(requireContext(), "file not found...!!!", Toast.LENGTH_LONG).show();
                }
            });
        }
    }

    public void downloadVideoWithDelay(long delayMillis) {
        requireActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        videoView.setVisibility(View.VISIBLE);
                        downloadVideo();
                    }
                }, delayMillis);
            }
        });
    }

}