package com.example.emotionrecognition

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.util.Pair
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.*
import java.util.*

class EmotionPyTorchClassifier(context: Context) {
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
        val index = arrayOfNulls<Int>(scores.size)
        for (i in scores.indices) {
            index[i] = i
        }
        Arrays.sort(
            index
        ) { idx1, idx2 -> java.lang.Float.compare(scores[idx2!!], scores[idx1!!]) }
        val K = 3
        val str = StringBuilder()
        str.append("Timecost (ms):").append(java.lang.Long.toString(res.first))
            .append("\nResult:\n")
        for (i in 0 until K) {
            str.append(
                "${labels!![index[i]!!]} ${index[i].toString()} ${scores[index[i]!!]}"
            )
        }
        Log.i(TAG, "PyTorch result: $str")
        return labels!![index[0]!!]
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

    init {
        module = Module.load(assetFilePath(context, MODEL_FILE))
        loadLabels(context)
    }
}