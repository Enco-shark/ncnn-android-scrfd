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
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

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

        // 初始化Handler用于更新Canvas
        updateHandler = new Handler(Looper.getMainLooper());
        updateRunnable = new Runnable() {
            @Override
            public void run() {
                if (cameraView != null && cameraView.getHolder() != null) {
                    drawChineseText(cameraView.getHolder(), cameraView.getWidth(), cameraView.getHeight());
                }
                // 每500ms更新一次
                updateHandler.postDelayed(this, 500);
            }
        };
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

        // 在surface上绘制中文文字
        drawChineseText(holder, width, height);
    }

    private void drawChineseText(SurfaceHolder holder, int width, int height)
    {
        Surface surface = holder.getSurface();
        if (surface == null || !surface.isValid()) {
            return;
        }

        Canvas canvas = null;
        try {
            canvas = surface.lockCanvas(null);
            if (canvas != null) {
                // 清除之前的绘制
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

                // 设置画笔
                android.graphics.Paint paint = new android.graphics.Paint();
                paint.setColor(Color.RED);
                paint.setTextSize(40);
                paint.setAntiAlias(true);
                paint.setStyle(android.graphics.Paint.Style.FILL);

                // 获取识别结果并绘制
                for (int i = 0; i < 10; i++) {  // 最多显示10个识别结果
                    String name = scrfdncnn.getRecognitionResult(i);
                    float similarity = scrfdncnn.getRecognitionSimilarity(i);

                    if (name != null && !name.isEmpty()) {
                        // 绘制人名
                        canvas.drawText(name, 50, 100 + i * 80, paint);

                        // 绘制相似度
                        paint.setColor(Color.GREEN);
                        paint.setTextSize(30);
                        String similarityText = String.format("%.1f%%", similarity * 100);
                        canvas.drawText(similarityText, 50, 130 + i * 80, paint);

                        // 恢复红色画笔
                        paint.setColor(Color.RED);
                        paint.setTextSize(40);
                    }
                }

                // 如果没有识别结果，显示提示文字
                if (scrfdncnn.getRecognitionResult(0) == null || scrfdncnn.getRecognitionResult(0).isEmpty()) {
                    canvas.drawText("等待人脸识别...", 50, 100, paint);
                }
            }
        } catch (Exception e) {
            Log.e("MainActivity", "Error drawing on surface: " + e.getMessage());
        } finally {
            if (canvas != null) {
                surface.unlockCanvasAndPost(canvas);
            }
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
        // 启动Canvas更新
        updateHandler.post(updateRunnable);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
        // 停止Canvas更新
        updateHandler.removeCallbacks(updateRunnable);
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
}
