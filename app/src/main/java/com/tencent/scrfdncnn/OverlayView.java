package com.tencent.scrfdncnn;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class OverlayView extends View {

    private static class FaceInfo {
        String name;
        float similarity;
        float rectX, rectY, rectW, rectH;
    }

    private final List<FaceInfo> faceList = new ArrayList<>();
    private final Paint textPaint = new Paint();
    private final Paint bgPaint = new Paint();
    private final Paint borderPaint = new Paint();
    private final Paint simPaint = new Paint();
    private final Paint simBgPaint = new Paint();
    private final Object lock = new Object();

    public OverlayView(Context context) {
        super(context);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        // 中文文本画笔
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(50f);
        textPaint.setAntiAlias(true);
        textPaint.setTypeface(Typeface.DEFAULT);  // 使用系统默认字体，支持中文

        // 背景画笔
        bgPaint.setColor(Color.argb(180, 255, 255, 255));  // 半透明白色背景
        bgPaint.setStyle(Paint.Style.FILL);

        // 边框画笔
        borderPaint.setColor(Color.BLACK);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(3f);

        // 相似度文本画笔
        simPaint.setColor(Color.GREEN);
        simPaint.setTextSize(36f);
        simPaint.setAntiAlias(true);

        // 相似度背景
        simBgPaint.setColor(Color.argb(180, 255, 255, 255));
        simBgPaint.setStyle(Paint.Style.FILL);
    }

    public void updateFaces(List<String> names, List<Float> similarities,
                            List<Float> rectXs, List<Float> rectYs,
                            List<Float> rectWs, List<Float> rectHs) {
        synchronized (lock) {
            faceList.clear();
            int count = Math.min(names.size(), Math.min(similarities.size(),
                    Math.min(rectXs.size(), Math.min(rectYs.size(),
                            Math.min(rectWs.size(), rectHs.size())))));
            for (int i = 0; i < count; i++) {
                FaceInfo info = new FaceInfo();
                info.name = names.get(i);
                info.similarity = similarities.get(i);
                info.rectX = rectXs.get(i);
                info.rectY = rectYs.get(i);
                info.rectW = rectWs.get(i);
                info.rectH = rectHs.get(i);
                faceList.add(info);
            }
        }
        postInvalidate();  // 触发重绘
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        synchronized (lock) {
            for (FaceInfo face : faceList) {
                if (face.name == null || face.name.isEmpty()) continue;

                // 计算文本位置（在人脸框上方）
                float textX = face.rectX;
                float textY = face.rectY - 10;

                // 测量文本宽度
                float textWidth = textPaint.measureText(face.name);
                float textHeight = textPaint.getTextSize();
                float simText = simPaint.measureText(String.format("%.1f%%", face.similarity * 100));
                float simHeight = simPaint.getTextSize();

                float bgWidth = Math.max(textWidth, simText) + 20;
                float bgHeight = textHeight + simHeight + 20;

                // 确保不超出边界
                if (textX + bgWidth > getWidth()) {
                    textX = getWidth() - bgWidth - 5;
                }
                if (textX < 0) textX = 0;
                if (textY - bgHeight < 0) {
                    textY = face.rectY + face.rectH + bgHeight;
                }
                if (textY > getHeight()) textY = getHeight() - 5;

                // 绘制半透明背景
                canvas.drawRoundRect(
                        new RectF(textX, textY - bgHeight, textX + bgWidth, textY + 5),
                        8, 8, bgPaint);

                // 绘制边框
                canvas.drawRoundRect(
                        new RectF(textX, textY - bgHeight, textX + bgWidth, textY + 5),
                        8, 8, borderPaint);

                // 绘制中文人名
                canvas.drawText(face.name, textX + 10, textY - simHeight - 5, textPaint);

                // 绘制相似度
                canvas.drawText(String.format("%.1f%%", face.similarity * 100),
                        textX + 10, textY - 5, simPaint);
            }
        }
    }
}
