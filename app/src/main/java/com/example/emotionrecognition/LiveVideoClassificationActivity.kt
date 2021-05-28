package com.example.emotionrecognition
//
//import android.graphics.*
//import android.media.Image
//import android.os.SystemClock
//import android.view.TextureView
//import android.view.ViewStub
//import android.widget.TextView
//import androidx.annotation.WorkerThread
//import androidx.camera.core.ImageProxy
//import org.pytorch.IValue
//import org.pytorch.Module
//import org.pytorch.PyTorchAndroid
//import org.pytorch.Tensor
//import org.pytorch.torchvision.TensorImageUtils
//import java.io.ByteArrayOutputStream
//import java.nio.FloatBuffer
//import java.util.*
//
//
//abstract class LiveVideoClassificationActivity :
//    AbstractCameraXActivity<LiveVideoClassificationActivity.AnalysisResult?>() {
//    private var mModule: Module? = null
//    private var mResultView: TextView? = null
//    private var mFrameCount = 0
//    private var inTensorBuffer: FloatBuffer? = null
//
//    class AnalysisResult(private val mResults: String)
//
//    override val contentViewLayoutId: Int
//        protected get() = R.layout.activity_live_video_classification
//    override val cameraPreviewTextureView: TextureView
//        protected get() {
//            mResultView = findViewById(R.id.resultView)
//            return (findViewById<ViewStub>(R.id.object_detection_texture_view_stub))
//                .inflate()
//                .findViewById(R.id.object_detection_texture_view)
//        }
//
//    protected fun applyToUiAnalyzeImageResult(result: AnalysisResult) {
//        mResultView!!.text = result.mResults
//        mResultView!!.invalidate()
//    }
//
//    private fun imgToBitmap(image: Image): Bitmap {
//        val planes = image.planes
//        val yBuffer = planes[0].buffer
//        val uBuffer = planes[1].buffer
//        val vBuffer = planes[2].buffer
//        val ySize = yBuffer.remaining()
//        val uSize = uBuffer.remaining()
//        val vSize = vBuffer.remaining()
//        val nv21 = ByteArray(ySize + uSize + vSize)
//        yBuffer[nv21, 0, ySize]
//        vBuffer[nv21, ySize, vSize]
//        uBuffer[nv21, ySize + vSize, uSize]
//        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
//        val out = ByteArrayOutputStream()
//        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 75, out)
//        val imageBytes = out.toByteArray()
//        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
//    }
//
//    @WorkerThread
//    protected fun analyzeImage(image: ImageProxy, rotationDegrees: Int): AnalysisResult? {
//        if (mModule == null) {
//            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "video_classification.pt")
//        }
//        if (mFrameCount == 0) inTensorBuffer = Tensor.allocateFloatBuffer(Constants.MODEL_INPUT_SIZE)
//        var bitmap = imgToBitmap(image.image!!)
//        val matrix = Matrix()
//        matrix.postRotate(90.0f)
//        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
//        val ratio = Math.min(bitmap.width, bitmap.height) / 160.0f
//        val resizedBitmap = Bitmap.createScaledBitmap(
//            bitmap,
//            (bitmap.width / ratio).toInt(), (bitmap.height / ratio).toInt(), true
//        )
//        val centerCroppedBitmap = Bitmap.createBitmap(
//            resizedBitmap,
//            if (resizedBitmap.width > resizedBitmap.height) (resizedBitmap.width - resizedBitmap.height) / 2 else 0,
//            if (resizedBitmap.height > resizedBitmap.width) (resizedBitmap.height - resizedBitmap.width) / 2 else 0,
//            Constants.TARGET_VIDEO_SIZE, Constants.TARGET_VIDEO_SIZE
//        )
//        TensorImageUtils.bitmapToFloatBuffer(
//            centerCroppedBitmap,
//            0,
//            0,
//            Constants.TARGET_VIDEO_SIZE,
//            Constants.TARGET_VIDEO_SIZE,
//            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
//            TensorImageUtils.TORCHVISION_NORM_STD_RGB,
//            inTensorBuffer,
//            Constants.MODEL_INPUT_SIZE * mFrameCount)
//        mFrameCount++
//        if (mFrameCount < 4) {
//            return null
//        }
//        mFrameCount = 0
//        val inputTensor = Tensor.fromBlob(
//            inTensorBuffer, longArrayOf(
//                Constants.COUNT_OF_FRAMES_PER_INFERENCE.toLong(),
//                3, //channels
//                Constants.TARGET_FACE_SIZE.toLong(), Constants.TARGET_FACE_SIZE.toLong()
//            )
//        )
//        val startTime = SystemClock.elapsedRealtime()
//        val outputTensor = mModule!!.forward(IValue.from(inputTensor)).toTensor()
//        val inferenceTime = SystemClock.elapsedRealtime() - startTime
//        val scores = outputTensor.dataAsFloatArray
//        val scoresIdx = arrayOfNulls<Int>(scores.size)
//        for (i in scores.indices) scoresIdx[i] = i
//        Arrays.sort(
//            scoresIdx
//        ) { o1, o2 -> java.lang.Float.compare(scores[o2!!], scores[o1!!]) }
//        val tops = arrayOfNulls<String>(Constants.TOP_COUNT)
//        for (j in 0 until Constants.TOP_COUNT) tops[j] = MainActivity.getClasses().get(scoresIdx[j])
//        val result = java.lang.String.join(", ", *tops)
//        return AnalysisResult(String.format("%s - %dms", result, inferenceTime))
//    }
//}