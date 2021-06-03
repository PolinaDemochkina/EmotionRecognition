package com.example.emotionrecognition

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.TextureView
import android.view.View
import android.view.View.GONE
import android.view.View.VISIBLE
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.UiThread
import androidx.annotation.WorkerThread
import androidx.camera.core.*
import androidx.camera.core.Preview.OnPreviewOutputUpdateListener
import androidx.camera.core.Preview.PreviewOutput
import androidx.core.app.ActivityCompat
import com.example.emotionrecognition.EmotionPyTorchVideoClassifier.Companion.applyToUiAnalyzeImageResult
import com.example.emotionrecognition.mtcnn.Box
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.*
import kotlin.time.ExperimentalTime

class  CameraActivity : BaseModuleActivity() {
    private val REQUEST_CODE_CAMERA_PERMISSION = 200
    private val PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    private var mLastAnalysisResultTime: Long = 0
    private var mFrameCount = 0
    private var inTensorBuffer: FloatBuffer? = null

    private fun getContentViewLayoutId(): Int {
        return R.layout.activity_live_video_classification
    }

    private fun getCameraPreviewTextureView(): TextureView {
        return findViewById(R.id.object_detection_texture_view)
    }

    @ExperimentalTime
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(getContentViewLayoutId())
        startBackgroundThread()
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                PERMISSIONS,
                REQUEST_CODE_CAMERA_PERMISSION
            )
        } else {
            setupCameraX()
        }
    }

    @ExperimentalTime
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(
                    this,
                    "You can't use live video classification example without granting CAMERA permission",
                    Toast.LENGTH_LONG
                )
                    .show()
                finish()
            } else {
                setupCameraX()
            }
        }
    }

    @ExperimentalTime
    @SuppressLint("RestrictedApi")
    fun setupCameraX() {
        val textureView = getCameraPreviewTextureView()
        val previewConfig = PreviewConfig.Builder()
            .setLensFacing(CameraX.LensFacing.FRONT)
            .build()
        val preview = Preview(previewConfig)
        var size: Size? = null
        preview.onPreviewOutputUpdateListener =
            OnPreviewOutputUpdateListener { output: PreviewOutput ->
                textureView.setSurfaceTexture(
                    output.surfaceTexture
                )
                size = output.textureSize
            }

        val imageAnalysisConfig = ImageAnalysisConfig.Builder()
            .setLensFacing(CameraX.LensFacing.FRONT)
            .setCallbackHandler(mBackgroundHandler)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build()

        val imageAnalysis = ImageAnalysis(imageAnalysisConfig)
        imageAnalysis.analyzer =
            ImageAnalysis.Analyzer { image: ImageProxy?, rotationDegrees: Int ->
                if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime > 500) {
                    runOnUiThread {
                        val overlay: ImageView = findViewById(com.example.emotionrecognition.R.id.overlay)
                        overlay.setImageResource(android.R.color.transparent)
                        val text: TextView = findViewById(R.id.liveText)
                        text.visibility = VISIBLE
                        text.text = "NO FACES DETECTED"
                    }
                }
                val height = size!!.height
                val width = size!!.width
                val result = analyzeImage(image, height, width)
                if (result != null) {
                    mLastAnalysisResultTime = SystemClock.elapsedRealtime()
                    runOnUiThread {
                        val text: TextView = findViewById(R.id.liveText)
                        text.visibility = GONE
                        applyToUiAnalyzeImageResult(result, height, width, findViewById(R.id.overlay))}
                }
            }

        CameraX.unbindAll()
        CameraX.bindToLifecycle(this, preview, imageAnalysis)
    }

    @ExperimentalTime
    @WorkerThread
    fun analyzeImage(image: ImageProxy?, width: Int, height: Int): EmotionPyTorchVideoClassifier.AnalysisResult? {
        if (mFrameCount == 0) inTensorBuffer =
            Tensor.allocateFloatBuffer(Constants.MODEL_INPUT_SIZE*Constants.COUNT_OF_FRAMES_PER_INFERENCE)

        var bitmap = imgToBitmap(image!!.image!!)
        val matrix = Matrix()
        matrix.postRotate(270.0f)
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
        val resizedBitmap = MainActivity.resize(bitmap, false)
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
        var bboxOrig: Rect?
        if (MainActivity.videoDetector != null &&  bbox != null) {
            bboxOrig = Rect(
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

        return EmotionPyTorchVideoClassifier.AnalysisResult(
            Rect(
                width - (width * bbox!!.left / resizedBitmap.width),
                height * bbox.top / resizedBitmap.height,
                width - (width * bbox.right / resizedBitmap.width),
                height * bbox.bottom / resizedBitmap.height
            ), result, width, height
        )
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

    fun back(view: View) {
        finish()
    }
}
