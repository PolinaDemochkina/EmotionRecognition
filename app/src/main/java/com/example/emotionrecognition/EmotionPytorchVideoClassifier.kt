package com.example.emotionrecognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.os.SystemClock
import android.util.Log
import android.util.Pair
import com.example.emotionrecognition.mtcnn.Box
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jvm.JvmMath.maxD2
import org.jetbrains.kotlinx.multik.jvm.JvmMath.minD2
import org.jetbrains.kotlinx.multik.jvm.JvmStatistics.meanD2
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.*
import java.util.*
import kotlin.collections.ArrayList


class EmotionPyTorchVideoClassifier(context: Context) {
    companion object {
        /** Tag for the [Log].  */
        private const val TAG = "Video detection"
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
        val br: BufferedReader?
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

    fun recognize(fromMs: Int,
                  toMs: Int,
                  mmr: MediaMetadataRetriever): String {
        val length = 1280
        val res = classifyVideo(fromMs, toMs, mmr).second
        val scores = mutableListOf<Float>()
        for (i in 0 until Constants.COUNT_OF_FRAMES_PER_INFERENCE){
            if ((i+1)*length <= res.size) {
                scores.addAll(res.sliceArray(length*i until length*(i+1)).toList())
            }
        }
        val features = mk.ndarray(mk[scores])
        val min = minD2(features, axis = 0).toList()
        val max = maxD2(features, axis = 0).toList()
        val mean: List<Float> = meanD2(features, axis = 0).toList().map { it.toFloat() }
        val std = mutableListOf<Float>()
        val rows = features.shape[0]
        for (i in 0 until length) {
            std.add(calculateSD(features[0.r..rows, i].toList()))
        }
        val descriptor = mean + std + min + max // scores + scores + scores + scores   // mean + std + min + max
        val index = MainActivity.clf?.predict(descriptor)
        Log.e(MainActivity.TAG, index.toString())
        return labels!![index!!]
    }

    private fun calculateSD(numArray: List<Float>): Float {
        var sum = 0.0
        var standardDeviation = 0.0

        for (num in numArray) {
            sum += num
        }

        val mean = sum / numArray.size

        for (num in numArray) {
            standardDeviation += Math.pow(num - mean, 2.0)
        }

        val divider = numArray.size - 1

        return Math.sqrt(standardDeviation / divider).toFloat()
    }

    private fun classifyVideo( fromMs: Int,
                               toMs: Int,
                               mmr: MediaMetadataRetriever ): Pair<Long, FloatArray> {
        val inTensorBuffer = Tensor.allocateFloatBuffer(Constants.MODEL_INPUT_SIZE)

        // extract 4 frames for each second of the video and pack them to a float buffer to be converted to the model input tensor
        for (i in 0 until Constants.COUNT_OF_FRAMES_PER_INFERENCE) {
            val timeUs =
                (1000 * (fromMs + ((toMs - fromMs) * i / (Constants.COUNT_OF_FRAMES_PER_INFERENCE - 1.0)).toInt())).toLong()
            val bitmap = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            val resizedBitmap = SecondActivity.resize(bitmap)
            val bboxes: Vector<Box> = MainActivity.mtcnnFaceDetector!!.detectFaces(
                resizedBitmap!!,
                MainActivity.minFaceSize
            )

            val box: Box? = bboxes.maxByOrNull { box ->
                box.score
            }

            val bbox = box?.transform2Rect()
            if (MainActivity.imageDetector != null && bbox!!.width() > 0 && bbox!!.height() > 0) {
                val bboxOrig = Rect(
                    bbox!!.left * resizedBitmap.width / resizedBitmap.width,
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

                TensorImageUtils.bitmapToFloatBuffer(
                    faceBitmap,
                    0,
                    0,
                    Constants.TARGET_VIDEO_SIZE,
                    Constants.TARGET_VIDEO_SIZE,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                    inTensorBuffer,
                    (Constants.COUNT_OF_FRAMES_PER_INFERENCE - 1) * i * Constants.TARGET_VIDEO_SIZE * Constants.TARGET_VIDEO_SIZE
                )
            }
        }

        val inputTensor = Tensor.fromBlob(
            inTensorBuffer, longArrayOf(
                Constants.COUNT_OF_FRAMES_PER_INFERENCE.toLong(),
                channels.toLong(),
                Constants.TARGET_VIDEO_SIZE.toLong(), Constants.TARGET_VIDEO_SIZE.toLong()
            )
        )
        val startTime = SystemClock.elapsedRealtime()
        val outputTensor: Tensor = module!!.forward(IValue.from(inputTensor)).toTensor()
        val timecostMs = SystemClock.uptimeMillis() - startTime
        Log.i(
            TAG,
            "Timecost to run PyTorch model inference: $timecostMs"
        )
        val scores = outputTensor.dataAsFloatArray
        Log.d(TAG, outputTensor.shape()[0].toString())
        Log.d(TAG, outputTensor.shape()[1].toString())
        return Pair(timecostMs, scores)
    }

    init {
        module = Module.load(assetFilePath(context, MODEL_FILE))
        loadLabels(context)
    }
}
