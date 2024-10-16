package com.iti.home_gp_test;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import android.database.Cursor;

import androidx.appcompat.app.AppCompatActivity;

public class login extends AppCompatActivity {

    DataBase DB;
    Button signInLogin;
    EditText userNameLogin, passwordLogin;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.login);

        userNameLogin = (EditText) findViewById(R.id.username);
        passwordLogin = (EditText) findViewById(R.id.password);
        signInLogin = (Button) findViewById(R.id.button);
        DB = new DataBase(this);

        signInLogin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String user = userNameLogin.getText().toString();
                String pass = passwordLogin.getText().toString();

                if (user.equals("") || pass.equals("")) {
                    Toast.makeText(login.this, "Please Enter All The Fields", Toast.LENGTH_SHORT).show();
                } else {
                    Boolean check = DB.checkusernamepassword(user, pass);
                    if (!check) {
                        Toast.makeText(login.this, "Make Sure Of Yours Input", Toast.LENGTH_SHORT).show();
                    } else {
                        Cursor cursor = DB.getUserData(user);
                        if (cursor.moveToFirst()) {
                            String gender = cursor.getString(cursor.getColumnIndex("gender"));
                            Toast.makeText(login.this, "Gender: " + gender, Toast.LENGTH_SHORT).show(); // Print gender value
                            if (gender.equals("female")) {
                                Intent intent = new Intent(login.this, choose_avatar_female.class);
                                intent.putExtra("Name", user);
                                startActivity(intent);
                            } else {
                                Intent intent = new Intent(login.this, choose_avatar_male.class);
                                intent.putExtra("Name", user);
                                startActivity(intent);
                            }
                        }
                        Toast.makeText(login.this, "Sign In Successfully", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        });

    }
}
