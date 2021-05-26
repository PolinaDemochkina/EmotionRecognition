package com.example.emotionrecognition

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.SystemClock
import android.provider.SyncStateContract
import android.util.Log
import android.util.Pair
import android.view.View
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.*
import java.util.*


class EmotionPyTorchVideoClassifier(context: Context) {
    companion object {
        /** Tag for the [Log].  */
        private const val TAG = "EmotionPyTorch"
        private const val MODEL_FILE = "mobile_efficientNet.pt"
        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String?): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }
            context.assets.open(assetName!!).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        }
    }

    private var labels: ArrayList<String>? = null
    private var module: Module? = null
    private val width = 224
    private val height = 224
    private val channels = 3
    private fun loadLabels(context: Context) {
        var br: BufferedReader? = null
        labels = ArrayList()
        try {
            br = BufferedReader(InputStreamReader(context.assets.open("affectnet_labels.txt")))
            br.useLines { lines -> lines.forEach {
                val categoryInfo = it.trim { it <= ' ' }.split(":").toTypedArray()
                val category = categoryInfo[1]
                labels!!.add(category) } }
            br.close()
        } catch (e: IOException) {
            throw RuntimeException("Problem reading emotion label file!", e)
        }
    }

    fun recognize(bitmap: Bitmap): String {
        val res = classifyImage(bitmap)
        val scores = res.second
//        val features = mk.ndarray(scores.asList())
//        val min = mk.math.minD2(features, axis = 1)
//        val max = mk.math.maxD2(features, axis = 1)
//        val mean = mk.stat.meanD2(features, axis = 1)
//        val std =
        val descriptor = scores + scores + scores + scores
        val index = MainActivity.clf?.predict(descriptor)
        return labels!![index!!]
    }

    private fun classifyImage(bitmap: Bitmap): Pair<Long, FloatArray> {
        var bitmap: Bitmap? = bitmap
        bitmap = Bitmap.createScaledBitmap(bitmap!!, width, height, false)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        val startTime = SystemClock.uptimeMillis()
        val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
        val timecostMs = SystemClock.uptimeMillis() - startTime
        Log.i(
            TAG,
            "Timecost to run PyTorch model inference: $timecostMs"
        )
        val scores = outputTensor.dataAsFloatArray
        Log.d(MainActivity.TAG, scores.size.toString())
        return Pair(timecostMs, scores)
    }

//    @Override
//    fun run() {
//        val mmr = MediaMetadataRetriever()
//        mmr.setDataSource(this.applicationContext, MainActivity.content)
//        val stringDuration = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
//        val durationMs = stringDuration!!.toDouble()
//
//        // for each second of the video, make inference to get the class label
//        val durationTo = Math.ceil(durationMs / 1000).toInt()
//        mResults.clear()
//        var i = 0
//        while (!mStopThread && i < durationTo) {
//            val from = i * 1000
//            var to = (i + 1) * 1000
//            if (i == durationTo - 1) to = Math.ceil(durationMs).toInt() - i * 1000
//            val pair: Pair<Array<Int>, Long> = getResult(from, to, mmr)
//            val scoresIdx = pair.first
//            val tops = arrayOfNulls<String>(Constants.TOP_COUNT)
//            for (j in 0 until Constants.TOP_COUNT) tops[j] = MainActivity.mClasses.get(scoresIdx[j])
//            val result = java.lang.String.join(", ", *tops)
//            val inferenceTime = pair.second
//            if (i * 1000 > mVideoView!!.currentPosition) {
//                try {
//                    Thread.sleep((i * 1000 - mVideoView!!.currentPosition).toLong())
//                } catch (e: InterruptedException) {
//                    Log.e(MainActivity.TAG, "Thread sleep exception: " + e.localizedMessage)
//                }
//            }
//            while (!mVideoView!!.isPlaying) {
//                if (mStopThread || mVideoView!!.currentPosition >= mVideoView!!.duration) break
//                try {
//                    Thread.sleep(100)
//                } catch (e: InterruptedException) {
//                    Log.e(MainActivity.TAG, "Thread sleep exception: " + e.localizedMessage)
//                }
//            }
//            val finalI = i
//            runOnUiThread {
//                mTextView.setVisibility(View.VISIBLE)
//                mTextView.setText(
//                    String.format(
//                        "%ds: %s - %dms",
//                        finalI + 1,
//                        result,
//                        inferenceTime
//                    )
//                )
//            }
//            mResults.add(result)
//            i++
//        }
//    }
//
//    private fun getResult(
//        fromMs: Int,
//        toMs: Int,
//        mmr: MediaMetadataRetriever
//    ): Pair<Array<Int?>, Long>? {
//        val inTensorBuffer = Tensor.allocateFloatBuffer(Constants.MODEL_INPUT_SIZE)
//
//        // extract 4 frames for each second of the video and pack them to a float buffer to be converted to the model input tensor
//        for (i in 0 until Constants.COUNT_OF_FRAMES_PER_INFERENCE) {
//            val timeUs =
//                (1000 * (fromMs + ((toMs - fromMs) * i / (Constants.COUNT_OF_FRAMES_PER_INFERENCE - 1.0)).toInt())).toLong()
//            val bitmap = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
//            val ratio =
//                Math.min(bitmap!!.width, bitmap.height) / Constants.TARGET_VIDEO_SIZE.toFloat()
//            val resizedBitmap = Bitmap.createScaledBitmap(
//                bitmap,
//                (bitmap.width / ratio).toInt(), (bitmap.height / ratio).toInt(), true
//            )
//            val centerCroppedBitmap = Bitmap.createBitmap(
//                resizedBitmap,
//                if (resizedBitmap.width > resizedBitmap.height) (resizedBitmap.width - resizedBitmap.height) / 2 else 0,
//                if (resizedBitmap.height > resizedBitmap.width) (resizedBitmap.height - resizedBitmap.width) / 2 else 0,
//                Constants.TARGET_VIDEO_SIZE, Constants.TARGET_VIDEO_SIZE
//            )
//            TensorImageUtils.bitmapToFloatBuffer(
//                centerCroppedBitmap,
//                0,
//                0,
//                Constants.TARGET_VIDEO_SIZE,
//                Constants.TARGET_VIDEO_SIZE,
//                Constants.MEAN_RGB,
//                Constants.STD_RGB,
//                inTensorBuffer,
//                (Constants.COUNT_OF_FRAMES_PER_INFERENCE - 1) * i * Constants.TARGET_VIDEO_SIZE * Constants.TARGET_VIDEO_SIZE
//            )
//        }
//        val inputTensor = Tensor.fromBlob(
//            inTensorBuffer, longArrayOf(
//                1, 3,
//                Constants.COUNT_OF_FRAMES_PER_INFERENCE.toLong(), 160, 160
//            )
//        )
//        val startTime = SystemClock.elapsedRealtime()
//        val outputTensor: Tensor = mModule.forward(IValue.from(inputTensor)).toTensor()
//        val inferenceTime = SystemClock.elapsedRealtime() - startTime
//        val scores = outputTensor.dataAsFloatArray
//        val scoresIdx = arrayOfNulls<Int>(scores.size)
//        for (i in scores.indices) scoresIdx[i] = i
//        Arrays.sort(
//            scoresIdx
//        ) { o1, o2 -> java.lang.Float.compare(scores[o2!!], scores[o1!!]) }
//        return Pair(scoresIdx, inferenceTime)
//    }

    init {
        module = Module.load(assetFilePath(context, MODEL_FILE))
        loadLabels(context)
    }
}