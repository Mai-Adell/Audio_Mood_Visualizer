package com.iti.home_gp_test;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class sign_up extends AppCompatActivity {

    Button signUpPage;
    DataBase DB;
    EditText userNameSignUp, passwordSignUP,phone,gender,age;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.sign_up);

        userNameSignUp=(EditText) findViewById(R.id.username2);
        passwordSignUP=(EditText) findViewById(R.id.password1);
        phone=(EditText) findViewById(R.id.Phone);
        gender=(EditText) findViewById(R.id.Gender);
        age=(EditText) findViewById(R.id.AGE);
        signUpPage=(Button) findViewById(R.id.sign_master);

        DB=new DataBase(this);


        signUpPage.setOnClickListener(view -> {
            String user=userNameSignUp.getText().toString();
            String pass=passwordSignUP.getText().toString();
            String PHONE=phone.getText().toString();
            String GENDER=gender.getText().toString();
            String AGE=age.getText().toString();


            if(user.equals("")||pass.equals("")||PHONE.equals("")||GENDER.equals("")||AGE.equals("")){
                Toast.makeText(sign_up.this,"Please Enter All The Fields",Toast.LENGTH_SHORT).show();
            }else {
                Boolean checkUser = DB.checkusername(user);
                if (checkUser) {
                    Toast.makeText(sign_up.this, "Please Change Username", Toast.LENGTH_SHORT).show();
                }else{
                    Boolean checkInsert=DB.insertData(user,pass,GENDER,PHONE,AGE);
                    if(checkInsert){
                        Toast.makeText(sign_up.this, "Registered Successfully", Toast.LENGTH_SHORT).show();
                        Intent intent=new Intent(sign_up.this,login.class);
                        startActivity(intent);
                    }
                    else{
                        Toast.makeText(sign_up.this, "Error", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        });
    }}