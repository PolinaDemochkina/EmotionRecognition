package com.example.emotionrecognition

import android.annotation.SuppressLint
import android.content.ContentResolver
import android.graphics.*
import android.media.ExifInterface
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.example.emotionrecognition.mtcnn.Box
import kotlinx.android.synthetic.main.activity_second.*
import java.util.*
import kotlin.math.ceil
import kotlin.time.ExperimentalTime


class SecondActivity : Runnable, AppCompatActivity() {
    companion object {
        fun resize(frame: Bitmap?, image: Boolean): Bitmap? {
            var ratio: Float
            if (image) {
                ratio = Math.min(frame!!.width, frame.height) / Constants.TARGET_IMAGE_SIZE.toFloat()
            }
            else {
                ratio =Math.min(frame!!.width, frame.height) / Constants.TARGET_VIDEO_SIZE.toFloat()
            }

            return Bitmap.createScaledBitmap(
                frame,
                (frame.width / ratio).toInt(),
                (frame.height / ratio).toInt(),
                false
            )
        }
    }

    private var mThread: Thread? = Thread(this)
    private var mStopThread = true

    @SuppressLint("WrongThread")
    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_second)

        val cR: ContentResolver = this.applicationContext.contentResolver
        val type = cR.getType(MainActivity.content!!)
        if (type == "image/jpeg") {
            Photo.visibility = View.VISIBLE
            imageRecognition()
        } else if (type == "video/mp4") {
            video.visibility = View.VISIBLE
            startVideo()
        }
    }

    private fun startVideo() {
        video?.setVideoURI(MainActivity.content)
        video?.setOnCompletionListener {
            TopPanel.visibility = View.VISIBLE
            Back.visibility = View.VISIBLE
        }
        video?.start()
        if (mThread != null && mThread!!.isAlive) {
            try {
                mThread!!.join()
            } catch (e: InterruptedException) {
                Log.e(MainActivity.TAG, e.localizedMessage)
            }
        }
        mStopThread = false
        mThread?.start()
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
    private fun imageRecognition() {
        val resizedBitmap: Bitmap? = resize(getImage(MainActivity.content!!), true)
        val bboxes: Vector<Box> = MainActivity.mtcnnFaceDetector!!.detectFaces(
            resizedBitmap!!,
            Constants.MIN_FACE_SIZE
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
            if (MainActivity.videoDetector != null && bbox.width() > 0 && bbox.height() > 0) {
                val faceBitmap = Bitmap.createBitmap(
                    resizedBitmap,
                    bbox.left,
                    bbox.top,
                    bbox.width(),
                    bbox.height()
                )
                val res: String = MainActivity.videoDetector!!.recognizeImage(faceBitmap)
                c.drawText(res, bbox.left.toFloat(), Math.max(0, bbox.top - 20).toFloat(), p_text)
                Log.i(MainActivity.TAG, res)
            }
        }
        Photo.setImageBitmap(tempBmp)
        TopPanel.visibility = View.VISIBLE
        Back.visibility = View.VISIBLE
    }

    private fun stopVideo() {
        video.stopPlayback()
        mStopThread = true
    }

    override fun onPause() {
        super.onPause()
        video.pause()
    }

    override fun onStop() {
        super.onStop()
        stopVideo()
    }

    @ExperimentalTime
    override fun run() {   // video recognition
        val mmr = MediaMetadataRetriever()
        mmr.setDataSource(this.applicationContext, MainActivity.content)
        val stringDuration = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
        val durationMs = stringDuration!!.toDouble()

        // for each second of the video, make inference to get the class label
        val durationTo = durationMs.toInt()
        var from = 0
        while (!mStopThread && from < durationTo) {
            from += 1000
            var to = from + 1000
            if (to > durationTo) to = ceil(durationMs).toInt()

            val start = System.nanoTime()
            val result: String = MainActivity.videoDetector!!.recognizeVideo(from, to, mmr)
            val elapsed = (System.nanoTime() - start)/10000000
            Log.e(MainActivity.TAG, String.format("Timecost to run PyTorch model inference: $elapsed"))
            Log.e(MainActivity.TAG, result)

            from += elapsed.toInt()

            if (from > video.currentPosition) {
                try {
                    Thread.sleep((from - video!!.currentPosition).toLong())
                } catch (e: InterruptedException) {
                    Log.e(MainActivity.TAG, "Thread sleep exception: " + e.localizedMessage)
                }
            }
            while (!video!!.isPlaying) {
                if (mStopThread || video!!.currentPosition >= video!!.duration) break
                try {
                    Thread.sleep(100)
                } catch (e: InterruptedException) {
                    Log.e(MainActivity.TAG, "Thread sleep exception: " + e.localizedMessage)
                }
            }

            runOnUiThread {
                detectionResult.visibility = View.VISIBLE
                detectionResult?.text = String.format(result)
            }
        }
    }

    fun back(view: View) {
        finish()
    }
}