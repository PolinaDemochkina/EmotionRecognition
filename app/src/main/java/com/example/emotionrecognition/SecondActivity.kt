package com.example.emotionrecognition

import android.animation.ObjectAnimator
import android.annotation.SuppressLint
import android.content.ContentResolver
import android.graphics.*
import android.media.ExifInterface
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Pair
import android.view.View
import android.widget.TextView
import android.widget.VideoView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.animation.doOnEnd
import com.example.emotionrecognition.mtcnn.Box
import kotlinx.android.synthetic.main.activity_second.*
import org.pytorch.IValue
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.util.*


class SecondActivity : AppCompatActivity() {
    private var mThread: Thread? = null
    private var mVideoView: VideoView? = null
    private var mStopThread = false
    private val mResults: List<String> = ArrayList()
    private var mTextView: TextView? = null

    @SuppressLint("WrongThread")
    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_second)
        mVideoView = findViewById(R.id.videoView)
//        mTextView = findViewById(R.id.textView)
//        mTextView.setVisibility(View.INVISIBLE)

        val cR: ContentResolver = this.applicationContext.contentResolver
        val type = cR.getType(MainActivity.content!!)
        if (type == "image/jpeg") {
            val resizedBitmap = resize(getImage(MainActivity.content!!))
            if (resizedBitmap != null) Photo.setImageBitmap(resizedBitmap)
            ProgressBar.max = 100
        } else if (type == "video/mp4") {
            startVideo()
//            val mmr = FFmpegMediaMetadataRetriever()
//            mmr.setDataSource(this.applicationContext, MainActivity.content)
//            val duration: Long = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_DURATION).toLong()
//            val frameRate: Double = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_FRAMERATE).toDouble()
//            val sec: Long = (1000 * 1000 / (frameRate)).roundToLong() // От этого!!!
//            val bitmaps: ArrayList<Bitmap> = arrayListOf()
//
//            for (i in 0..duration * 1000 step sec) {
//                val bitmap: Bitmap?  = mmr.getFrameAtTime(i, FFmpegMediaMetadataRetriever.OPTION_CLOSEST)
//                try {
//                    if (bitmap != null) {
//                        bitmaps.add(resize(bitmap)!!)
//                    }
//                } catch (e : java.lang.Exception) {
//                    e.printStackTrace();
//                }
//            }
        }
    }

    private fun startVideo() {
        mVideoView?.setVideoURI(MainActivity.content)
        mVideoView?.start()
        if (mThread != null && mThread!!.isAlive()) {
            try {
                mThread!!.join()
            } catch (e: InterruptedException) {
                Log.e(MainActivity.TAG, e.localizedMessage)
            }
        }
//        mStopThread = false
//        mThread = Thread(this@SecondActivity)
//        mThread?.start()
    }

    private fun resize(frame: Bitmap?): Bitmap? {
        var resizedBitmap = frame
        val minSize = 600.0
        val scale = Math.min(resizedBitmap!!.width, resizedBitmap.height) / minSize
        if (scale > 1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(
                frame!!,
                (frame.width / scale).toInt(),
                (frame.height / scale).toInt(), false
            )
        }
        return resizedBitmap
    }

    @RequiresApi(Build.VERSION_CODES.N)
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
        }
        progress.start()
        mtcnnDetectionAndEmotionPyTorchRecognition()
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

    @RequiresApi(Build.VERSION_CODES.N)
    private fun mtcnnDetectionAndEmotionPyTorchRecognition() {
        val resizedBitmap: Bitmap? = resize(getImage(MainActivity.content!!))
        val startTime = SystemClock.uptimeMillis()
        val bboxes: Vector<Box> = MainActivity.mtcnnFaceDetector!!.detectFaces(
            resizedBitmap!!,
            MainActivity.minFaceSize
        ) //(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(
            MainActivity.TAG,
            "Timecost to run mtcnn: " + java.lang.Long.toString(SystemClock.uptimeMillis() - startTime)
        )
        val tempBmp = Bitmap.createBitmap(resizedBitmap.width, resizedBitmap.height, Bitmap.Config.ARGB_8888)
        val c = Canvas(tempBmp)
        val p = Paint()
        p.style = Paint.Style.STROKE
        p.isAntiAlias = true
        p.isFilterBitmap = true
        p.isDither = true
        p.color = Color.parseColor("#9FFFCB")
        p.strokeWidth = 5f
        val p_text = Paint()
        p_text.color = Color.WHITE
        p_text.style = Paint.Style.FILL
        p_text.color = Color.parseColor("#9FFFCB")
        p_text.textSize = 24f
        c.drawBitmap(resizedBitmap, 0f, 0f, null)
        for (box in bboxes) {
            val bbox =
                box.transform2Rect() //new android.graphics.Rect(Math.max(0,box.left()),Math.max(0,box.top()),box.right(),box.bottom());
            p.color = Color.parseColor("#9FFFCB")
            c.drawRect(bbox, p)
            if (MainActivity.emotionClassifierPyTorch != null && bbox.width() > 0 && bbox.height() > 0) {
                val bboxOrig = Rect(
                    bbox.left * resizedBitmap.width / resizedBitmap.width,
                    resizedBitmap.height * bbox.top / resizedBitmap.height,
                    resizedBitmap.width * bbox.right / resizedBitmap.width,
                    resizedBitmap.height * bbox.bottom / resizedBitmap.height
                )
                val faceBitmap = Bitmap.createBitmap(
                    resizedBitmap,
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