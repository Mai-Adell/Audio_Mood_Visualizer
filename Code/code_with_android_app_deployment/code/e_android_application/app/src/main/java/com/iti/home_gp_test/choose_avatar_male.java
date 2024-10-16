package com.iti.home_gp_test;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class choose_avatar_male extends AppCompatActivity {


    private ImageView smallGirlImageView;
    private ImageView middleGirlImageView;
    private ImageView oldestGirlImageView;
    private String name;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_choose_avatar_male);

        Intent intent = getIntent();
        name = intent.getStringExtra("Name");


        smallGirlImageView = findViewById(R.id.small_girl);
        middleGirlImageView = findViewById(R.id.middle_girl);
        oldestGirlImageView = findViewById(R.id.oldest_girl);

        smallGirlImageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openAnotherActivity(1);
            }
        });

        middleGirlImageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openAnotherActivity(2);
            }
        });

        oldestGirlImageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openAnotherActivity(3);
            }
        });
    }

    private void openAnotherActivity(int pageNumber) {
        Intent intent = new Intent(this, MainActivity.class);
        intent.putExtra("page_number", pageNumber);
        intent.putExtra("gender", "boy");
        intent.putExtra("Name", name);
        startActivity(intent);
    }
}
