package com.iti.home_gp_test;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class DataBase extends SQLiteOpenHelper {
    public static final String DBName="LoginData.db";
    public DataBase(Context context) {
        super(context, "LoginData.db", null, 1);
    }

    @Override
    public void onCreate(SQLiteDatabase MyDB) {
        MyDB.execSQL("create Table users(username TEXT primary key ,password TEXT,gender TEXT ,phone TEXT,age TEXT)");
    }
    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        //String sql = "ALTER TABLE " + "users" + " ADD COLUMN " +
        //       "age" + " INTEGER";
        db.execSQL("ALTER TABLE users ADD COLUMN email TEXT");
        //db.execSQL(sql);
    }
   /* @Override
    public void onUpgrade(SQLiteDatabase MyDB, int i, int i1) {
      MyDB.execSQL("drop Table if exists users");
    }*/

    public  Boolean insertData(String username,String password,String gender,String phone,String age){
        SQLiteDatabase MyDB=this.getWritableDatabase();
        ContentValues contentValues=new ContentValues();
        contentValues.put("username",username);
        contentValues.put("password",password);
        contentValues.put("gender",gender);
        contentValues.put("phone",phone);
        contentValues.put("age", age);
        long result=MyDB.insert("users",null,contentValues);

        if(result==-1){
            return false;
        }else{
            return true;
        }
    }
    public Boolean checkusername(String username){
        SQLiteDatabase MyDB=this.getWritableDatabase();
        Cursor cursor=MyDB.rawQuery("select * from users where username =?",new String[]{username});
        if(cursor.getCount()>0){
            return true;
        }else{
            return false;
        }
    }
    public Boolean checkusernamepassword(String username,String password){
        SQLiteDatabase MyDB=this.getWritableDatabase();
        Cursor cursor=MyDB.rawQuery("select * from users where username =? and password=?",new String[]{username,password});
        if(cursor.getCount()>0){
            return true;
        }else{
            return false;
        }
    }
    public Cursor getUserData(String username) {
        SQLiteDatabase MyDB = this.getWritableDatabase();
        return MyDB.rawQuery("select * from users where username =?", new String[]{username});
    }
}
