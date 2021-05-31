package com.example.emotionrecognition
//
//import android.annotation.SuppressLint
//import android.media.MediaMetadataRetriever
//import android.os.Bundle
//import android.os.SystemClock
//import android.util.Log
//import android.view.View
//import android.widget.ImageView
//import android.widget.VideoView
//import kotlinx.android.synthetic.main.activity_second.*
//import kotlin.math.ceil
//import kotlin.time.ExperimentalTime
//
//class  GalleryVideoActivity : BaseModuleActivity() {
//    private var mLastAnalysisResultTime: Long = 0
//
//    @ExperimentalTime
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        startBackgroundThread()
//        setContentView(R.layout.activity_second)
//
//        video?.setVideoURI(MainActivity.content)
//        video?.setOnCompletionListener {
//            TopPanel.visibility = View.VISIBLE
//            Back.visibility = View.VISIBLE
//        }
//        setupCameraX()
//    }
//
//    @ExperimentalTime
//    @SuppressLint("RestrictedApi")
//    fun setupCameraX() {
//        video?.start()
//        val mmr = MediaMetadataRetriever()
//        mmr.setDataSource(this.applicationContext, MainActivity.content)
//        val stringDuration = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
//        val durationMs = stringDuration!!.toDouble()
//        val height = mmr.extractMetadata((MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT))
//        val width = mmr.extractMetadata((MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH))
//
//        // for each second of the video, make inference to get the class label
//        val durationTo = durationMs.toInt()
//        var from = 0
//        while (from < durationTo) {
//            from += 1000
//            var to = from + 1000
//            if (to > durationTo) to = ceil(durationMs).toInt()
//
//            val start = System.nanoTime()
//            val result = MainActivity.videoDetector!!.recognizeVideo(from, to, mmr)
//            val elapsed = (System.nanoTime() - start)/10000000
//            Log.e(MainActivity.TAG, String.format("Timecost to run PyTorch model inference: $elapsed"))
//
////            from += elapsed.toInt()
//
//            if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime > 250) {
//                runOnUiThread {
//                    val overlay: ImageView = findViewById(R.id.gallery_overlay)
//                    overlay.setImageResource(android.R.color.transparent)
//                }
//            }
//
//            if (result != null) {
//                mLastAnalysisResultTime = SystemClock.elapsedRealtime()
//                runOnUiThread { applyToUiAnalyzeImageResult(result, width!!.toInt(), height!!.toInt(), findViewById(R.id.gallery_overlay))}
//            }
//        }
//    }
//
//    fun back(view: View) {
//        finish()
//    }
//}
