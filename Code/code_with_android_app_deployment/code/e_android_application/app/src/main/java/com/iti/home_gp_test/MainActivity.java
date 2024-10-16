package com.iti.home_gp_test;

import static android.content.ContentValues.TAG;

import static java.lang.Math.log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.PackageManagerCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;
import kotlin.Unit;
import kotlin.jvm.functions.Function1;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.Toast;

import com.etebarian.meowbottomnavigation.MeowBottomNavigation;

public class MainActivity extends AppCompatActivity {

    private MeowBottomNavigation bottom_navigation_bar;
    private LinearLayout home_layout, profile_layout, realTime_layout;
    Button uploadBtn, recordBtn;
    private String gender;

    private Drawable Background_uploadBtn, Background_recordBtn, Background_uploadBtn_pressed, Background_recordBtn_pressed;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Intent intent = getIntent();
        int page_number = intent.getIntExtra("page_number",0);
        gender = intent.getStringExtra("gender");
        String name = intent.getStringExtra("Name");


        //Toast.makeText(MainActivity.this, "Main_Genderrrrrrrr: " + gender, Toast.LENGTH_SHORT).show();
        //Toast.makeText(MainActivity.this, "Main_page_number: " + page_number, Toast.LENGTH_SHORT).show();


        home_layout = findViewById(R.id.home_page);
        profile_layout = findViewById(R.id.profile);
        realTime_layout = findViewById(R.id.real_time);

        bottom_navigation_bar = findViewById(R.id.bottom_navigation_bar);

        if (bottom_navigation_bar != null) {

            MeowBottomNavigation.Model item1 = new MeowBottomNavigation.Model(1,R.drawable.real_time_img);
            MeowBottomNavigation.Model item2 = new MeowBottomNavigation.Model(2,R.drawable.profile_img);
            MeowBottomNavigation.Model item3 = new MeowBottomNavigation.Model(3,R.drawable.home_img);

            bottom_navigation_bar.add(item1);
            bottom_navigation_bar.add(item2);
            bottom_navigation_bar.add(item3);

            profile_layout.setVisibility(View.VISIBLE);
            home_layout.setVisibility(View.GONE);
            realTime_layout.setVisibility(View.GONE);

            bottom_navigation_bar.setOnClickMenuListener(new MeowBottomNavigation.ClickListener() {

                @Override
                public void onClickItem(MeowBottomNavigation.Model item) {

                    //Toast.makeText(MainActivity.this,"cliked item " + item.getId(),Toast.LENGTH_SHORT).show();

                    if(item.getId() == 1)
                    {
                        //Toast.makeText(MainActivity.this,"in 1" ,Toast.LENGTH_SHORT).show();
                        profile_layout.setVisibility(View.GONE);
                        home_layout.setVisibility(View.GONE);
                        realTime_layout.setVisibility(View.VISIBLE);

                    }
                    else if(item.getId() == 2)
                    {
                        //Toast.makeText(MainActivity.this,"in 2" ,Toast.LENGTH_SHORT).show();
                        profile_layout.setVisibility(View.VISIBLE);
                        home_layout.setVisibility(View.GONE);
                        realTime_layout.setVisibility(View.GONE);

                    }
                    else if(item.getId() == 3)
                    {
                        //Toast.makeText(MainActivity.this,"in 3" ,Toast.LENGTH_SHORT).show();
                        profile_layout.setVisibility(View.GONE);
                        home_layout.setVisibility(View.VISIBLE);
                        realTime_layout.setVisibility(View.GONE);
                        home_upload_fragment fragment = home_upload_fragment.newInstance(gender,page_number);
                        replaceFragment(fragment);

                    }
                }
            });

            bottom_navigation_bar.setOnShowListener(new MeowBottomNavigation.ShowListener() {
                @Override
                public void onShowItem(MeowBottomNavigation.Model item) {

                  //  Toast.makeText(MainActivity.this,"showed item " + item.getId(),Toast.LENGTH_SHORT).show();

                }
            });

            bottom_navigation_bar.setCount(2,"1");
            bottom_navigation_bar.show(2,true);

            // ...
        } else {
            Log.e("MainActivity", "MeowBottomNavigation is null");
        }
        //home page
        uploadBtn = findViewById(R.id.btn_upload);
        recordBtn = findViewById(R.id.btn_rec);

        Background_uploadBtn = ContextCompat.getDrawable(this, R.drawable.upload_btn_layout);//uploadBtn.getBackground();
        Background_recordBtn = ContextCompat.getDrawable(this, R.drawable.record_btn_layout);//recordBtn.getBackground();

        Background_uploadBtn_pressed = ContextCompat.getDrawable(this, R.drawable.upload_btn_pressedon_layout);//uploadBtn.getBackground();
        Background_recordBtn_pressed = ContextCompat.getDrawable(this, R.drawable.record_btn_pressedon_layout);//recordBtn.getBackground();

        uploadBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //recordBtn.setBackgroundTintList(ColorStateList.valueOf(getResources().getColor(android.R.color.holo_blue_light)));
                recordBtn.setBackground(Background_recordBtn_pressed);
                recordBtn.setTextColor(getResources().getColor(R.color.black));
                uploadBtn.setBackground(Background_uploadBtn);
                uploadBtn.setTextColor(getResources().getColor(R.color.white));
                home_upload_fragment fragment = home_upload_fragment.newInstance(gender,page_number);
                replaceFragment(fragment);
            }
        });


        recordBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //uploadBtn.setBackgroundTintList(ColorStateList.valueOf(getResources().getColor(android.R.color.holo_green_light)));
                uploadBtn.setBackground(Background_uploadBtn_pressed);
                uploadBtn.setTextColor(getResources().getColor(R.color.black));
                recordBtn.setBackground(Background_recordBtn);
                recordBtn.setTextColor(getResources().getColor(R.color.white));
                replaceFragment(new home_record_fragment());
            }
        });


        //profile page

        EditText profileName = findViewById(R.id.profileName);
        ImageView profileImage = findViewById(R.id.profileImage);

        profileName.setText(name);

        Log.d("TAG", gender);
        if(gender.equals("girl"))
        {
            switch (page_number)
            {
                case 1:
                    profileImage.setImageResource(R.drawable.small_girl);
                    break;
                case 2:
                    profileImage.setImageResource(R.drawable.middle_girl);
                    break;
                case 3:
                    profileImage.setImageResource(R.drawable.oldest_girl);
                    break;
            }
        }
        else
        {
            switch (page_number)
            {
                case 1:
                    profileImage.setImageResource(R.drawable.small_boy);
                    break;
                case 2:
                    profileImage.setImageResource(R.drawable.middle_boy);
                    break;
                case 3:
                    profileImage.setImageResource(R.drawable.oldest_boy);
                    break;
            }
        }

    }

    private void replaceFragment(Fragment fragment) {

        FragmentManager fragmentManager = getSupportFragmentManager();
        FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();
        fragmentTransaction.replace(R.id.fragmentHomeView,fragment);
        fragmentTransaction.commit();
    }


}