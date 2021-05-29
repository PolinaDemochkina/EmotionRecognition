package com.example.emotionrecognition

import android.graphics.*
import android.media.Image
import android.util.Log
import android.view.TextureView
import android.view.View
import android.view.ViewStub
import android.widget.TextView
import androidx.annotation.WorkerThread
import androidx.camera.core.ImageProxy
import com.example.emotionrecognition.mtcnn.Box
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.*
import kotlin.time.ExperimentalTime


class LiveVideoClassificationActivity :
    AbstractCameraXActivity<LiveVideoClassificationActivity.AnalysisResult?>() {
    private var mResultView: TextView? = null
    private var mFrameCount = 0
    private var inTensorBuffer: FloatBuffer? = null

    class AnalysisResult(val mResults: String)

    override fun getContentViewLayoutId(): Int {
        return R.layout.activity_live_video_classification
    }

    override fun getCameraPreviewTextureView(): TextureView {
        mResultView = findViewById(R.id.resultView)
        return (findViewById<ViewStub>(R.id.object_detection_texture_view_stub))
            .inflate()
            .findViewById(R.id.object_detection_texture_view)
    }

    override fun applyToUiAnalyzeImageResult(result: AnalysisResult?) {
        mResultView!!.text = result!!.mResults
        mResultView!!.invalidate()
    }

    private fun imgToBitmap(image: Image): Bitmap {
        val planes = image.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer[nv21, 0, ySize]
        vBuffer[nv21, ySize, vSize]
        uBuffer[nv21, ySize + vSize, uSize]
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 75, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    @ExperimentalTime
    @WorkerThread
    override fun analyzeImage(image: ImageProxy?, rotationDegrees: Int): AnalysisResult? {
        if (mFrameCount == 0) inTensorBuffer =
            Tensor.allocateFloatBuffer(Constants.MODEL_INPUT_SIZE*Constants.COUNT_OF_FRAMES_PER_INFERENCE)

        var bitmap = imgToBitmap(image!!.image!!)
        val matrix = Matrix()
        matrix.postRotate(90.0f)
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
        val resizedBitmap = SecondActivity.resize(bitmap, false)
        val start = System.nanoTime()
        val bboxes: Vector<Box> = MainActivity.mtcnnFaceDetector!!.detectFaces(
            resizedBitmap!!,
            Constants.MIN_FACE_SIZE
        )
        val elapsed = (System.nanoTime() - start)/10000000
        Log.e(MainActivity.TAG, "Timecost to run MTCNN: $elapsed")

        val box: Box? = bboxes.maxByOrNull { box ->
            box.score
        }

        val bbox = box?.transform2Rect()
        if (MainActivity.videoDetector != null &&  bbox != null) {
            val bboxOrig = Rect(
                bitmap.width * bbox.left / resizedBitmap.width,
                bitmap.height * bbox.top / resizedBitmap.height,
                bitmap.width * bbox.right / resizedBitmap.width,
                bitmap.height * bbox.bottom / resizedBitmap.height
            )
            val face = Bitmap.createScaledBitmap(Bitmap.createBitmap(bitmap,
                bboxOrig.left,
                bboxOrig.top,
                bboxOrig.width(),
                bboxOrig.height()),
                Constants.TARGET_FACE_SIZE, Constants.TARGET_FACE_SIZE, false)

            TensorImageUtils.bitmapToFloatBuffer(
                face,
                0,
                0,
                Constants.TARGET_FACE_SIZE,
                Constants.TARGET_FACE_SIZE,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                inTensorBuffer,
                (mFrameCount * Constants.MODEL_INPUT_SIZE))

            mFrameCount++
        }

        if (mFrameCount < Constants.COUNT_OF_FRAMES_PER_INFERENCE) {
            return null
        }

        mFrameCount = 0

        val result = MainActivity.videoDetector!!.recognizeLiveVideo(inTensorBuffer!!)

        return AnalysisResult(String.format("%s", result))
    }

    fun back(view: View) {
        finish()
    }
}