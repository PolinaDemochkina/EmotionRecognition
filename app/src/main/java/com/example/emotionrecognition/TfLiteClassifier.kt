package com.example.emotionrecognition

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*

abstract class TfLiteClassifier(context: Context?, model_path: String?) {
    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    protected var tflite: Interpreter

    /* Preallocated buffers for storing image data in. */
    private var intValues: IntArray? = null
    protected lateinit var imgData: ByteBuffer

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.  */
    var imageSizeX = 224
    var imageSizeY = 224
    private val outputs: Array<Array<FloatArray?>>
    var outputMap: MutableMap<Int, Any> = HashMap()
    protected abstract fun addPixelValue(`val`: Int)

    /** Classifies a frame from the preview stream.  */
    fun classifyFrame(bitmap: Bitmap): ClassifierResult? {
        val inputs = arrayOf<Any?>(null)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        if (imgData == null) {
            return null
        }
        imgData!!.rewind()
        // Convert the image to floating point.
        var pixel = 0
        for (i in 0 until imageSizeX) {
            for (j in 0 until imageSizeY) {
                val `val` = intValues!![pixel++]
                addPixelValue(`val`)
            }
        }
        inputs[0] = imgData
        val startTime = SystemClock.uptimeMillis()
        tflite.runForMultipleInputsOutputs(inputs, outputMap)
        for (i in outputs.indices) {
            val ith_output = outputMap[i] as ByteBuffer?
            ith_output!!.rewind()
            val len: Int = outputs[i][0]!!.size
            for (j in 0 until len) {
                outputs[i][0]!![j] = ith_output.float
            }
            ith_output.rewind()
        }
        val endTime = SystemClock.uptimeMillis()
        Log.i(
            TAG,
            "tf lite timecost to run model inference: " + (endTime - startTime).toString()
        )
        return getResults(outputs)
    }

    fun close() {
        tflite.close()
    }

    protected abstract fun getResults(outputs: Array<Array<FloatArray?>>?): ClassifierResult?

    // Float.SIZE / Byte.SIZE;
    protected val numBytesPerChannel: Int
        protected get() = 4 // Float.SIZE / Byte.SIZE;

    companion object {
        /** Tag for the [Log].  */
        private const val TAG = "TfLiteClassifier"
    }

    init {
        //GpuDelegate delegate = new GpuDelegate();
        val options = Interpreter.Options().setNumThreads(4) //.addDelegate(delegate);
        if (false) {
            val opt = GpuDelegate.Options()
            opt.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
            val delegate = GpuDelegate()
            options.addDelegate(delegate)
        }
        val tfliteModel = FileUtil.loadMappedFile(context!!, model_path!!)
        tflite = Interpreter(tfliteModel, options)
        tflite.allocateTensors()
        val inputShape = tflite.getInputTensor(0).shape()
        imageSizeX = inputShape[1]
        imageSizeY = inputShape[2]
        intValues = IntArray(imageSizeX * imageSizeY)
        imgData =
            ByteBuffer.allocateDirect(imageSizeX * imageSizeY * inputShape[3] * numBytesPerChannel)
        imgData.order(ByteOrder.nativeOrder())
        val outputCount = tflite.outputTensorCount
        outputs = Array(outputCount) { arrayOfNulls(1) }
        for (i in 0 until outputCount) {
            val shape = tflite.getOutputTensor(i).shape()
            val numOFFeatures = shape[1]
            Log.i(TAG, "Read output layer size is $numOFFeatures")
            outputs[i][0] = FloatArray(numOFFeatures)
            val ith_output =
                ByteBuffer.allocateDirect(numOFFeatures * numBytesPerChannel) // Float tensor, shape 3x2x4
            ith_output.order(ByteOrder.nativeOrder())
            outputMap[i] = ith_output
        }
    }
}