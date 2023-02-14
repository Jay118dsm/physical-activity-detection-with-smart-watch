package com.example.uiforwearabledevices;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
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
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import com.example.uiforwearabledevices.databinding.ActivityMainBinding;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import java.util.Set;
import java.util.UUID;

public class MainActivity extends Activity implements View.OnClickListener, SensorEventListener {


    private ActivityMainBinding binding;
    private Button btn_StartAndStop;
    private TextView tv_data1;
    private TextView tv_data2;
    private TextView tv_data3;
    private SensorManager mSensorMgr;
    private Chronometer chronometer;
    private Toast toast;
    private static final String TAG = "MY_APP_DEBUG_TAG";
    private Handler handler;
    private static BluetoothSocket socket;
    private BluetoothAdapter adapter;
    private Set<BluetoothDevice> device;
    static String acc;
    static String gyr;
    private static final UUID MY_UUID = UUID.fromString("fa87c0d0-afac-11de-8a39-0800200c9a66");


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
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        btn_StartAndStop = findViewById(R.id.btn_StartAndStop);
        btn_StartAndStop.setOnClickListener(this);
        tv_data1 = findViewById(R.id.tv_data1);
        tv_data2 = findViewById(R.id.tv_data2);
        tv_data3 = findViewById(R.id.tv_data3);
        mSensorMgr = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        chronometer = findViewById(R.id.chronometer);
        adapter = BluetoothAdapter.getDefaultAdapter();

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return;
        }
        device = adapter.getBondedDevices();
        if(device.isEmpty()){
            toast = Toast.makeText(this, "no connected device", Toast.LENGTH_SHORT);
            toast.show();
        }
        for(BluetoothDevice bluetoothDevice : device){
            boolean isconnected = false;
            try{
                isconnected = (boolean) bluetoothDevice.getClass().getMethod("isConnected").invoke(bluetoothDevice);
            } catch (InvocationTargetException|IllegalAccessException|NoSuchMethodException e) {
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
        ConnectedThread ct1 = new ConnectedThread(socket);
        ct1.run();
        String click = new String(ct1.mmBuffer);
        if(click.equals("click")){
            btn_StartAndStop.callOnClick();
        }



    }

    protected void onPause() {
        super.onPause();
        mSensorMgr.unregisterListener(this);
    }

    protected void onResume() {
        super.onResume();

    }

    protected void onStop() {
        super.onStop();
        mSensorMgr.unregisterListener(this);

    }

    @SuppressLint({"SetTextI18n", "DefaultLocale"})
    public void onSensorChanged(SensorEvent event) {
        ConnectedThread ct = new ConnectedThread(socket);
        int type = event.sensor.getType();
        float[] values = event.values;
        switch (type) {
            case Sensor.TYPE_ACCELEROMETER:
                acc = "1"+ Float.toString(values[0]) + "/" + Float.toString(values[1]) + "/" + Float.toString(values[2]);
                tv_data1.setText("X=" + String.format("%.2f", values[0]) + " Y=" + String.format("%.2f", values[1]) + " Z=" + String.format("%.2f", values[2]));
                ct.write(acc.getBytes(StandardCharsets.UTF_8));
                break;
            case Sensor.TYPE_GYROSCOPE:
                gyr = "2" + Float.toString(values[0]) + "/" + Float.toString(values[1]) + "/" + Float.toString(values[2]);
                tv_data2.setText("X=" + String.format("%.2f", values[0]) + " Y=" + String.format("%.2f", values[1]) + " Z=" + String.format("%.2f", values[2]));
                ct.write(gyr.getBytes(StandardCharsets.UTF_8));

                break;

        }

    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {

        return;
    }

    @Override
    public void startActivityForResult(Intent intent, int requestCode) {
        super.startActivityForResult(intent, requestCode);
    }

    @SuppressLint("SetTextI18n")
    @Override
    public void onClick(View view) {
        if (btn_StartAndStop.getText().equals("Start")) {
            toast = Toast.makeText(this, "START", Toast.LENGTH_SHORT);
            toast.show();
            chronometer.setVisibility(View.VISIBLE);
            chronometer.setBase(SystemClock.elapsedRealtime());
            chronometer.start();
            btn_StartAndStop.setText("Stop");
            btn_StartAndStop.setTextColor(Color.RED);
            btn_StartAndStop.setBackgroundColor(Color.GRAY);
            mSensorMgr.unregisterListener(this, mSensorMgr.getDefaultSensor(Sensor.TYPE_ACCELEROMETER));
            mSensorMgr.registerListener(this, mSensorMgr.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_UI);
            mSensorMgr.unregisterListener(this, mSensorMgr.getDefaultSensor(Sensor.TYPE_GYROSCOPE));
            mSensorMgr.registerListener(this, mSensorMgr.getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_UI);
            ConnectedThread ct1 = new ConnectedThread(socket);


        }
        else if(btn_StartAndStop.getText().equals("Stop")){
            toast = Toast.makeText(this, "STOP", Toast.LENGTH_SHORT);
            toast.show();
            chronometer.setVisibility(View.INVISIBLE);
            chronometer.setBase(SystemClock.elapsedRealtime());
            chronometer.stop();
            mSensorMgr.unregisterListener(this);
            tv_data1.setText("------");
            tv_data2.setText("------");
            tv_data3.setText("------");
            btn_StartAndStop.setText("Start");
            btn_StartAndStop.setTextColor(Color.BLACK);
            btn_StartAndStop.setBackgroundColor(Color.rgb(66,204,255));

        }

    }


}