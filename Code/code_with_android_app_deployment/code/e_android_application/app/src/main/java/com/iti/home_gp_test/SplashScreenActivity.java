package com.iti.home_gp_test;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import androidx.appcompat.app.AppCompatActivity;
import pl.droidsonroids.gif.GifImageView;

public class SplashScreenActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash_screen);

        GifImageView gifImageView = findViewById(R.id.gifImageView);
        gifImageView.setImageResource(R.drawable.splashimg);

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                // Start your main activity
                startActivity(new Intent(SplashScreenActivity.this, Start_Up.class));
                finish();
            }
        }, 3000); // 3000 milliseconds = 3 seconds
    }
}
