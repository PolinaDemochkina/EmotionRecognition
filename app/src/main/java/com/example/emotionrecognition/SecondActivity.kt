package com.example.emotionrecognition

import android.animation.ObjectAnimator
import android.graphics.*
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.animation.doOnEnd
import com.example.emotionrecognition.mtcnn.Box
import kotlinx.android.synthetic.main.activity_second.*
import java.util.*


class SecondActivity : AppCompatActivity() {
    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_second)
        MainActivity.sampledImage = getImage(MainActivity.image!!)
        var resizedBitmap = MainActivity.sampledImage
        val minSize = 600.0
        val scale = Math.min(resizedBitmap!!.width, resizedBitmap!!.height) / minSize
        if (scale > 1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(
                MainActivity.sampledImage!!,
                (MainActivity.sampledImage!!.width / scale).toInt(),
                (MainActivity.sampledImage!!.height / scale).toInt(), false
            )
        }
        if (resizedBitmap != null) Photo.setImageBitmap(resizedBitmap)
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
        if (isImageLoaded()) {
            mtcnnDetectionAndEmotionPyTorchRecognition()
        }
    }

    private fun isImageLoaded(): Boolean {
        if (MainActivity.sampledImage == null) Toast.makeText(
            applicationContext,
            "It is necessary to open image firstly",
            Toast.LENGTH_SHORT
        ).show()
        return MainActivity.sampledImage != null
    }
    @RequiresApi(Build.VERSION_CODES.N)
    private fun getImage(selectedImageUri: Uri): Bitmap? {
        var bmp: Bitmap? = null
        try {
            var ims = contentResolver.openInputStream(selectedImageUri)
            bmp = BitmapFactory.decodeStream(ims)
            ims!!.close()
            ims = contentResolver.openInputStream(selectedImageUri)
            val exif = ExifInterface(ims!!) //selectedImageUri.getPath());
            val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1)
            var degreesForRotation = 0
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> degreesForRotation = 90
                ExifInterface.ORIENTATION_ROTATE_270 -> degreesForRotation = 270
                ExifInterface.ORIENTATION_ROTATE_180 -> degreesForRotation = 180
            }
            if (degreesForRotation != 0) {
                val matrix = Matrix()
                matrix.setRotate(degreesForRotation.toFloat())
                bmp = Bitmap.createBitmap(
                    bmp!!, 0, 0, bmp.width,
                    bmp.height, matrix, true
                )
            }
        } catch (e: Exception) {
            Log.e(MainActivity.TAG, "Exception thrown: " + e + " " + Log.getStackTraceString(e))
        }
        return bmp
    }

    private fun mtcnnDetectionAndEmotionPyTorchRecognition() {
        var bmp: Bitmap = MainActivity.sampledImage!!
        var resizedBitmap = bmp
        val minSize = 600.0
        val scale = Math.min(bmp.width, bmp.height) / minSize
        if (scale > 1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(
                bmp,
                (bmp.width / scale).toInt(), (bmp.height / scale).toInt(), false
            )
            bmp = resizedBitmap
        }
        val startTime = SystemClock.uptimeMillis()
        val bboxes: Vector<Box> = MainActivity.mtcnnFaceDetector!!.detectFaces(
            resizedBitmap,
            MainActivity.minFaceSize
        ) //(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(
            MainActivity.TAG,
            "Timecost to run mtcnn: " + java.lang.Long.toString(SystemClock.uptimeMillis() - startTime)
        )
        val tempBmp = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
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
        c.drawBitmap(bmp, 0f, 0f, null)
        for (box in bboxes) {
            val bbox =
                box.transform2Rect() //new android.graphics.Rect(Math.max(0,box.left()),Math.max(0,box.top()),box.right(),box.bottom());
            p.color = Color.RED
            c.drawRect(bbox, p)
            if (MainActivity.emotionClassifierPyTorch != null && bbox.width() > 0 && bbox.height() > 0) {
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
                val res: String = MainActivity.emotionClassifierPyTorch!!.recognize(faceBitmap)
                c.drawText(res, bbox.left.toFloat(), Math.max(0, bbox.top - 20).toFloat(), p_text)
                Log.i(MainActivity.TAG, res)
            }
        }
        Photo.setImageBitmap(tempBmp)
    }

    fun back(view: View) {
        finish()
    }
}