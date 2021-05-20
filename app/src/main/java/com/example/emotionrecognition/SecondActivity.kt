package com.example.emotionrecognition

import android.animation.ObjectAnimator
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.animation.doOnEnd
import kotlinx.android.synthetic.main.activity_second.*
import java.util.*

class SecondActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_second)
        Photo.setImageURI(MainActivity.image)
        ProgressBar.max = 100
    }

    fun analyze(view: View) {
        Analyze.visibility = View.GONE
        AnalyzeText.visibility = View.VISIBLE
        ProgressBar.visibility = View.VISIBLE

        val progress = ObjectAnimator.ofInt(ProgressBar, "progress", 100)
        progress.setDuration(3000)
        progress.doOnEnd {
            AnalyzeText.visibility = View.INVISIBLE
            ProgressBar.visibility = View.INVISIBLE
            Success.visibility = View.VISIBLE
            TopPanel.visibility = View.VISIBLE
            Back.visibility = View.VISIBLE
            Download.visibility = View.VISIBLE
        }
        progress.start()

        mtcnnDetectionAndAttributesRecognition(null)
    }

    private fun mtcnnDetectionAndAttributesRecognition(classifier: TfLiteClassifier?) {
        val bmp: Bitmap = sampledImage
        var resizedBitmap = bmp
        val minSize = 600.0
        val scale = Math.min(bmp.width, bmp.height) / minSize
        if (scale > 1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(
                bmp,
                (bmp.width / scale).toInt(), (bmp.height / scale).toInt(), false
            )
            //bmp=resizedBitmap;
        }
        val startTime = SystemClock.uptimeMillis()
        val bboxes: Vector<Box> = mtcnnFaceDetector.detectFaces(
            resizedBitmap,
            MainActivity.minFaceSize
        ) //(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(
            MainActivity.TAG,
            "Timecost to run mtcnn: " + java.lang.Long.toString(SystemClock.uptimeMillis() - startTime)
        )
        val tempBmp =
            Bitmap.createBitmap(resizedBitmap.width, resizedBitmap.height, Bitmap.Config.ARGB_8888)
        val c = Canvas(tempBmp)
        val p = Paint()
        p.style = Paint.Style.STROKE
        p.isAntiAlias = true
        p.isFilterBitmap = true
        p.isDither = true
        p.color = Color.BLUE
        p.strokeWidth = 5f
        val p_text = Paint()
        p_text.color = Color.WHITE
        p_text.style = Paint.Style.FILL
        p_text.color = Color.BLUE
        p_text.textSize = 24f
        c.drawBitmap(resizedBitmap, 0f, 0f, null)
        for (box in bboxes) {
            p.color = Color.RED
            val bbox: Rect =
                box.transform2Rect() //new android.graphics.Rect(Math.max(0,box.left()),Math.max(0,box.top()),box.right(),box.bottom());
            c.drawRect(bbox, p)
            if (classifier != null && bbox.width() > 0 && bbox.height() > 0) {
                val bboxOrig = Rect(
                    bbox.left * bmp.width / resizedBitmap.width,
                    bmp.height * bbox.top / resizedBitmap.height,
                    bmp.width * bbox.right / resizedBitmap.width,
                    bmp.height * bbox.bottom / resizedBitmap.height
                )
                val faceBitmap = Bitmap.createBitmap(
                    bmp,
                    bboxOrig.left,
                    bboxOrig.top,
                    bboxOrig.width(),
                    bboxOrig.height()
                )
                val resultBitmap = Bitmap.createScaledBitmap(
                    faceBitmap,
                    classifier.imageSizeX,
                    classifier.imageSizeY,
                    false
                )
                val res: ClassifierResult = classifier.classifyFrame(resultBitmap)
                c.drawText(
                    res.toString(),
                    bbox.left.toFloat(),
                    Math.max(0, bbox.top - 20).toFloat(),
                    p_text
                )
                Log.i(MainActivity.TAG, res.toString())
            }
        }
        imageView.setImageBitmap(tempBmp)
    }

    fun back(view: View) {
        finish()
    }
}