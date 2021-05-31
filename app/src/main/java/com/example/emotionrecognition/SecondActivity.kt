package com.example.emotionrecognition

import android.annotation.SuppressLint
import android.content.ContentResolver
import android.graphics.*
import android.media.MediaMetadataRetriever
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_second.*
import kotlin.math.ceil
import kotlin.time.ExperimentalTime


class SecondActivity : Runnable, AppCompatActivity() {
    companion object {
        fun resize(frame: Bitmap?, image: Boolean): Bitmap? {
            val ratio: Float = if (image) {
                Math.min(frame!!.width, frame.height) / Constants.TARGET_IMAGE_SIZE.toFloat()
            } else {
                Math.min(frame!!.width, frame.height) / Constants.TARGET_VIDEO_SIZE.toFloat()
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
        if (type == "video/mp4") {
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

    fun back(view: View) {
        finish()
    }
}