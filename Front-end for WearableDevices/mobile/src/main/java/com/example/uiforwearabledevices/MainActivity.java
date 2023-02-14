package com.example.uiforwearabledevices;

import android.Manifest;
import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.icu.text.IDNA;
import android.icu.text.SimpleDateFormat;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.google.android.gms.common.api.PendingResult;
import com.google.android.gms.wearable.DataApi;
import com.google.android.gms.wearable.PutDataMapRequest;
import com.google.android.gms.wearable.PutDataRequest;
import com.google.android.gms.wearable.Wearable;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Date;
import java.util.EventListener;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.UUID;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private Button btn_StartAndStop;
    private Chronometer chronometer;
    private TextView tv_accX;
    private TextView tv_accY;
    private TextView tv_accZ;
    private TextView tv_gyrX;
    private TextView tv_gyrY;
    private TextView tv_gyrZ;
    private TextView tv_heart;
    private Toast toast;
    private Handler handler;
    private SensorManager mSensorMgr;
    private static BluetoothSocket socket;
    private BluetoothAdapter adapter;
    private Set<BluetoothDevice> device;
    public List<String> LS1;
    public List<String> LS2;
    public String acc;
    public String gyr;
    private static final String TAG = "MY_APP_DEBUG_TAG";
    private static final UUID MY_UUID = UUID.fromString("fa87c0d0-afac-11de-8a39-0800200c9a66");
    String read;


    @Override


    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},1);
        btn_StartAndStop = findViewById(R.id.btn_StartAndStop);
        chronometer = findViewById(R.id.chronometer);
        tv_accX = findViewById(R.id.tv_accX);
        tv_accY = findViewById(R.id.tv_accY);
        tv_accZ = findViewById(R.id.tv_accZ);
        tv_gyrX = findViewById(R.id.tv_gyrX);
        tv_gyrY = findViewById(R.id.tv_gyrY);
        tv_gyrZ = findViewById(R.id.tv_gyrZ);
        tv_heart = findViewById(R.id.tv_heart);
        btn_StartAndStop.setOnClickListener(this);
        mSensorMgr = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        LS1 = new ArrayList<String>();
        LS2 = new ArrayList<String>();
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.BLUETOOTH_CONNECT},1);// TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return;
        }

        device = adapter.getBondedDevices();
        toast = Toast.makeText(this, "succeed", Toast.LENGTH_SHORT);
        toast.show();
        if(device.isEmpty()){
            toast = Toast.makeText(this, "no connected device", Toast.LENGTH_SHORT);
            toast.show();
        }
        for(BluetoothDevice bluetoothDevice : device){
            boolean isconnected = false;
            try{
                isconnected = (boolean) bluetoothDevice.getClass().getMethod("isConnected").invoke(bluetoothDevice);
            } catch (InvocationTargetException |IllegalAccessException|NoSuchMethodException e) {
                toast = Toast.makeText(this, "can't get info", Toast.LENGTH_SHORT);
                toast.show();
            }
            if(isconnected){
                try {
                    socket = bluetoothDevice.createRfcommSocketToServiceRecord(MY_UUID);
                } catch (IOException e) {
                    toast = Toast.makeText(this, "can't get socket", Toast.LENGTH_SHORT);
                    toast.show();
                }
                break;
            }
        }

    }
    private interface MessageConstants {
        public static final int MESSAGE_READ = 0;
        public static final int MESSAGE_WRITE = 1;
        public static final int MESSAGE_TOAST = 2;
    }

    private class ConnectedThread extends Thread {
        private final BluetoothSocket mmSocket;
        private final InputStream mmInStream;
        private final OutputStream mmOutStream;
        private byte[] mmBuffer; // mmBuffer store for the stream

        public ConnectedThread(BluetoothSocket socket) {
            mmSocket = socket;
            InputStream tmpIn = null;
            OutputStream tmpOut = null;

            // Get the input and output streams; using temp objects because
            // member streams are final.
            try {
                tmpIn = socket.getInputStream();
            } catch (IOException e) {
                Log.e(TAG, "Error occurred when creating input stream", e);
            }
            try {
                tmpOut = socket.getOutputStream();
            } catch (IOException e) {
                Log.e(TAG, "Error occurred when creating output stream", e);
            }

            mmInStream = tmpIn;
            mmOutStream = tmpOut;
        }

        public void run() {
            mmBuffer = new byte[1024];
            int numBytes; // bytes returned from read()

            // Keep listening to the InputStream until an exception occurs.
            while (true) {
                try {
                    // Read from the InputStream.
                    numBytes = mmInStream.read(mmBuffer);
                    // Send the obtained bytes to the UI activity.
                    Message readMsg = handler.obtainMessage(
                            MessageConstants.MESSAGE_READ, numBytes, -1,
                            mmBuffer);
                    readMsg.sendToTarget();
                } catch (IOException e) {
                    Log.d(TAG, "Input stream was disconnected", e);
                    break;
                }
            }
        }

        // Call this from the main activity to send data to the remote device.
        public void write(byte[] bytes) {
            try {
                mmOutStream.write(bytes);

                // Share the sent message with the UI activity.
                Message writtenMsg = handler.obtainMessage(
                        MessageConstants.MESSAGE_WRITE, -1, -1, mmBuffer);
                writtenMsg.sendToTarget();
            } catch (IOException e) {
                Log.e(TAG, "Error occurred when sending data", e);

                // Send a failure message back to the activity.
                Message writeErrorMsg =
                        handler.obtainMessage(MessageConstants.MESSAGE_TOAST);
                Bundle bundle = new Bundle();
                bundle.putString("toast",
                        "Couldn't send data to the other device");
                writeErrorMsg.setData(bundle);
                handler.sendMessage(writeErrorMsg);
            }
        }

        // Call this method from the main activity to shut down the connection.
        public void cancel() {
            try {
                mmSocket.close();
            } catch (IOException e) {
                Log.e(TAG, "Could not close the connect socket", e);
            }
        }
    }






    @Override
    public void onClick(View v) {
        if(btn_StartAndStop.getText().equals("Start")){
            toast = Toast.makeText(this, "START", Toast.LENGTH_SHORT);
            toast.show();
            chronometer.setBase(SystemClock.elapsedRealtime());
            chronometer.start();
            btn_StartAndStop.setText("Stop");
            btn_StartAndStop.setTextColor(Color.RED);
            btn_StartAndStop.setBackgroundColor(Color.GRAY);

            ConnectedThread ct = new ConnectedThread(socket);
            String click = "click";
            ct.write(click.getBytes(StandardCharsets.UTF_8));
            ct.run();
            read = new String(ct.mmBuffer);
            if (read.length() < 10) {
                if (read.substring(0, 1).equals("1")) {
                    acc = read.substring(1);
                    String[] a = read.split("/");
                    tv_accX.setText(a[0]);
                    tv_accX.setText(a[1]);
                    tv_accX.setText(a[2]);
                    LS1.add(acc);
                } else if (read.substring(0, 1).equals("2")) {
                    gyr = read.substring(1);
                    String[] b = read.split("/");
                    tv_gyrX.setText(b[0]);
                    tv_gyrX.setText(b[1]);
                    tv_gyrX.setText(b[2]);
                    LS2.add(gyr);
                }
            }
        }
        else if(btn_StartAndStop.getText().equals("Stop")){
            toast = Toast.makeText(this, "STOP", Toast.LENGTH_SHORT);
            toast.show();
            chronometer.setBase(SystemClock.elapsedRealtime());
            chronometer.stop();
            tv_accX.setText("------");
            tv_accY.setText("------");
            tv_accZ.setText("------");
            tv_gyrX.setText("------");
            tv_gyrY.setText("------");
            tv_gyrZ.setText("------");
            tv_heart.setText("------");
            btn_StartAndStop.setText("Start");
            btn_StartAndStop.setTextColor(Color.BLACK);
            btn_StartAndStop.setBackgroundColor(Color.rgb(66,204,255));
            File filepath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),"Acc");

            if(!filepath.exists()){
                try{
                    filepath.mkdir();
                }catch (Exception e){
                    toast = Toast.makeText(this, "ERROR1", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
            int k = 0;
            while(true){
                String filename = filepath + "/Acc" + k +".txt";
                File file = new File(filename);
                if(!file.exists()){
                    try{
                        file.createNewFile();
                    }catch (Exception e){
                        toast = Toast.makeText(this, "ERROR2", Toast.LENGTH_SHORT);
                        toast.show();
                    }
                    try {
                        FileOutputStream fos = new FileOutputStream(file);
                        for(int i = 0;i< LS1.size();i++){
                            String content = LS1.get(i) + "\n";
                            fos.write(content.getBytes(StandardCharsets.UTF_8));
                        }
                        fos.close();
                        LS1 = new ArrayList<String>();
                    } catch (FileNotFoundException e) {
                        toast = Toast.makeText(this, "ERROR3", Toast.LENGTH_SHORT);
                        toast.show();
                    } catch (IOException e) {
                        toast = Toast.makeText(this, "ERROR4", Toast.LENGTH_SHORT);
                        toast.show();
                    }
                    break;
                }
                k++;
            }

            filepath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),"Gyr");

            if(!filepath.exists()){
                try{
                    filepath.mkdir();
                }catch (Exception e){
                    toast = Toast.makeText(this, "ERROR1", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }

            k = 0;
            while(true){
                String filename = filepath + "/Gyr" + k +".txt";
                File file = new File(filename);
                if(!file.exists()){
                    try{
                        file.createNewFile();
                    }catch (Exception e){
                        toast = Toast.makeText(this, "ERROR2", Toast.LENGTH_SHORT);
                        toast.show();
                    }
                    try {
                        FileOutputStream fos = new FileOutputStream(file);
                        for(int i = 0;i< LS2.size();i++){
                            String content = LS2.get(i) + "\n";
                            fos.write(content.getBytes(StandardCharsets.UTF_8));
                        }
                        fos.close();
                        LS2 = new ArrayList<String>();
                    } catch (FileNotFoundException e) {
                        toast = Toast.makeText(this, "ERROR3", Toast.LENGTH_SHORT);
                        toast.show();
                    } catch (IOException e) {
                        toast = Toast.makeText(this, "ERROR4", Toast.LENGTH_SHORT);
                        toast.show();
                    }
                    break;
                }
                k++;
            }




        }
    }
}
