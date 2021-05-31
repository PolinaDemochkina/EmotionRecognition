package com.example.emotionrecognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.util.Log
import android.util.Pair
import androidx.annotation.WorkerThread
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
import java.nio.FloatBuffer
import java.util.*
import kotlin.collections.ArrayList
import kotlin.time.ExperimentalTime


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
    private val length = 1280

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

    private fun classifyFeatures(res: FloatArray): String {
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
        val descriptor = mean + std + min + max
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

    @ExperimentalTime
    fun recognizeLiveVideo(inTensorBuffer: FloatBuffer): String {
        val res = getFeatures(inTensorBuffer)
        if (res != null) {
            return classifyFeatures(res)
        }
        return ""
    }

    private fun getFeatures(inTensorBuffer: FloatBuffer): FloatArray {
        val inputTensor = Tensor.fromBlob(
            inTensorBuffer, longArrayOf(
                Constants.COUNT_OF_FRAMES_PER_INFERENCE.toLong(),
                3, //channels
                Constants.TARGET_FACE_SIZE.toLong(), Constants.TARGET_FACE_SIZE.toLong()
            )
        )
        val outputTensor: Tensor = module!!.forward(IValue.from(inputTensor)).toTensor()

        val scores = outputTensor.dataAsFloatArray
        Log.d(TAG, outputTensor.shape()[0].toString())
        Log.d(TAG, outputTensor.shape()[1].toString())

        return scores
    }

    @ExperimentalTime
    @WorkerThread
    fun recognizeVideo(fromMs: Int,
                       toMs: Int,
                       mmr: MediaMetadataRetriever ): LiveVideoActivity.AnalysisResult? {
        var numFrames = 0
        var faces : MutableList<Bitmap> = mutableListOf()
        var bitmap: Bitmap? = null
        var resizedBitmap: Bitmap? = null
        var bbox: Rect? = null

        for (i in 0 until Constants.COUNT_OF_FRAMES_PER_INFERENCE) {
            val timeUs = (1000 * (fromMs + ((toMs - fromMs) * i /
                    (Constants.COUNT_OF_FRAMES_PER_INFERENCE - 1.0)).toInt())).toLong()
            bitmap = mmr.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            resizedBitmap = MainActivity.resize(bitmap, false)
            val start = System.nanoTime()
            val bboxes: Vector<Box> = MainActivity.mtcnnFaceDetector!!.detectFaces(
                resizedBitmap!!,
                Constants.MIN_FACE_SIZE
            )
            val elapsed = (System.nanoTime() - start)/10000000
            Log.e(TAG, "Timecost to run MTCNN: $elapsed")

            val box: Box? = bboxes.maxByOrNull { box ->
                box.score
            }

            bbox = box?.transform2Rect()
            if (MainActivity.videoDetector != null &&  bbox != null) {
                val bboxOrig = Rect(
                    bitmap!!.width * bbox.left / resizedBitmap.width,
                    bitmap.height * bbox.top / resizedBitmap.height,
                    bitmap.width * bbox.right / resizedBitmap.width,
                    bitmap.height * bbox.bottom / resizedBitmap.height
                )
                faces.add(Bitmap.createScaledBitmap(Bitmap.createBitmap(bitmap,
                    bboxOrig.left,
                    bboxOrig.top,
                    bboxOrig.width(),
                    bboxOrig.height()),
                    Constants.TARGET_FACE_SIZE, Constants.TARGET_FACE_SIZE, false))

                numFrames += 1
            }
        }

        if (numFrames > 0) {
            val inTensorBuffer = Tensor.allocateFloatBuffer(Constants.MODEL_INPUT_SIZE*numFrames)

            for (i in 0 until numFrames) {
                TensorImageUtils.bitmapToFloatBuffer(
                    faces[i],
                    0,
                    0,
                    Constants.TARGET_FACE_SIZE,
                    Constants.TARGET_FACE_SIZE,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                    inTensorBuffer,
                    (i * Constants.MODEL_INPUT_SIZE))
            }

            val features = getFeatures(inTensorBuffer)
            val emotion = classifyFeatures(features)

            return LiveVideoActivity.AnalysisResult(
                Rect(
                    (bitmap!!.width * bbox!!.left / resizedBitmap!!.width),
                    bitmap.height * bbox.top / resizedBitmap.height,
                    (bitmap.width * bbox.right / resizedBitmap.width),
                    bitmap.height * bbox.bottom / resizedBitmap.height
                ), emotion, bitmap.width, bitmap.height
            )
        }
        return null
    }

    init {
        module = Module.load(assetFilePath(context, MODEL_FILE))
        loadLabels(context)
    }
}
