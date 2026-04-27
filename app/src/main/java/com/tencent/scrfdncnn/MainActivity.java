// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.scrfdncnn;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;

    private SCRFDNcnn scrfdncnn = new SCRFDNcnn();
    private int facing = 0;

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;
    private OverlayView overlayView;
    private TextView recognitionResultView;
    private Handler updateHandler;
    private Runnable updateRunnable;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        overlayView = (OverlayView) findViewById(R.id.overlayView);
        recognitionResultView = (TextView) findViewById(R.id.recognitionResult);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                int new_facing = 1 - facing;

                // 关闭摄像头
                scrfdncnn.closeCamera();

                // 切换摄像头方向
                facing = new_facing;

                // 重新打开摄像头
                scrfdncnn.openCamera(facing);

                // 重新设置surface
                scrfdncnn.setOutputWindow(cameraView.getHolder().getSurface());
            }
        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        reload();

        // 初始化Handler用于更新识别结果显示
        updateHandler = new Handler(Looper.getMainLooper());
        updateRunnable = new Runnable() {
            @Override
            public void run() {
                if (recognitionResultView != null && overlayView != null) {
                    try {
                        updateRecognitionResults();
                    } catch (Exception e) {
                        Log.e("MainActivity", "Error updating results: " + e.getMessage());
                    }
                }
                // 每100ms更新一次识别结果显示（更快的刷新率）
                updateHandler.postDelayed(this, 100);
            }
        };
    }

    private void updateRecognitionResults()
    {
        StringBuilder resultText = new StringBuilder();
        boolean hasResult = false;

        try {
            int faceCount = scrfdncnn.getFaceCount();
            List<String> names = new ArrayList<>();
            List<Float> similarities = new ArrayList<>();
            List<Float> rectXs = new ArrayList<>();
            List<Float> rectYs = new ArrayList<>();
            List<Float> rectWs = new ArrayList<>();
            List<Float> rectHs = new ArrayList<>();

            for (int i = 0; i < faceCount; i++) {
                String name = scrfdncnn.getRecognitionResult(i);
                float similarity = scrfdncnn.getRecognitionSimilarity(i);
                float rx = scrfdncnn.getFaceRectX(i);
                float ry = scrfdncnn.getFaceRectY(i);
                float rw = scrfdncnn.getFaceRectWidth(i);
                float rh = scrfdncnn.getFaceRectHeight(i);

                if (name != null && !name.isEmpty()) {
                    names.add(name);
                    similarities.add(similarity);
                    rectXs.add(rx);
                    rectYs.add(ry);
                    rectWs.add(rw);
                    rectHs.add(rh);

                    if (hasResult) {
                        resultText.append("\n");
                    }
                    resultText.append(String.format("%s (%.1f%%)", name, similarity * 100));
                    hasResult = true;
                    Log.i("MainActivity", "Recognition: " + name + " -> " + (similarity * 100) + "%");
                }
            }

            // 更新 OverlayView（在摄像头画面上绘制中文）
            overlayView.updateFaces(names, similarities, rectXs, rectYs, rectWs, rectHs);

        } catch (Exception e) {
            Log.e("MainActivity", "Error reading recognition results: " + e.getMessage());
        }

        if (!hasResult) {
            recognitionResultView.setText("等待人脸识别...");
        } else {
            recognitionResultView.setText(resultText.toString());
        }
    }

    private void reload()
    {
        boolean ret_init = scrfdncnn.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "scrfdncnn loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        scrfdncnn.setOutputWindow(holder.getSurface());
    }


    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
        // 启动识别结果更新
        if (updateHandler != null && updateRunnable != null) {
            updateHandler.post(updateRunnable);
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
        // 停止识别结果更新
        if (updateHandler != null && updateRunnable != null) {
            updateHandler.removeCallbacks(updateRunnable);
        }
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        scrfdncnn.openCamera(facing);
    }

    @Override
    public void onPause()
    {
        super.onPause();

        scrfdncnn.closeCamera();
    }

    @Override
    public void onDestroy()
    {
        super.onDestroy();

        // 停止Handler的所有回调
        if (updateHandler != null) {
            if (updateRunnable != null) {
                updateHandler.removeCallbacks(updateRunnable);
                updateRunnable = null;
            }
            updateHandler = null;
        }

        // 关闭摄像头并清理资源
        if (scrfdncnn != null) {
            try {
                scrfdncnn.closeCamera();
                scrfdncnn.clearRecognitionResults();
            } catch (Exception e) {
                Log.e("MainActivity", "Error during cleanup: " + e.getMessage());
            }
        }

        cameraView = null;
        overlayView = null;
        recognitionResultView = null;

        Log.i("MainActivity", "onDestroy completed");
    }
}
