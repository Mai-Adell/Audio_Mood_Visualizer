package com.iti.home_gp_test;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class Start_Up extends AppCompatActivity {

    Button signInMain,signUPMain;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start_up);

        signInMain=(Button) findViewById(R.id.login2);
        signUPMain=(Button) findViewById(R.id.sign2);

        signUPMain.setOnClickListener(view -> {
            Intent intent=new Intent(Start_Up.this,sign_up.class);
            startActivity(intent);
        });

        signInMain.setOnClickListener(view -> {
            Intent intent=new Intent(Start_Up.this,login.class);
            startActivity(intent);
        });

    }
}