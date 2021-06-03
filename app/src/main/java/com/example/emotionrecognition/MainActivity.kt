package com.example.emotionrecognition

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.emotionrecognition.mtcnn.FaceDetector
import kotlinx.android.synthetic.main.activity_second.*
import java.util.*


class MainActivity : AppCompatActivity() {
    companion object {
        var content: Uri? = null
        const val TAG = "MainActivity"
        var mtcnnFaceDetector: FaceDetector? = null
        var videoDetector: EmotionPyTorchVideoClassifier? = null
        var clf: LinearSVC? = null

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

    private val REQUEST_ACCESS_TYPE = 1
    private val REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (!allPermissionsGranted()) {
            getRequiredPermissions()?.let {
                ActivityCompat.requestPermissions(
                    this,
                    it,
                    REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS
                )
            }
        }
        else
            init()
    }

    private fun getRequiredPermissions(): Array<String?>? {
        return try {
            val info = packageManager
                .getPackageInfo(packageName, PackageManager.GET_PERMISSIONS)
            val ps = info.requestedPermissions
            if (ps != null && ps.isNotEmpty()) {
                ps
            } else {
                arrayOfNulls(0)
            }
        } catch (e: java.lang.Exception) {
            arrayOfNulls(0)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS -> {
                val perms: MutableMap<String, Int> = HashMap()
                var allGranted = true
                var i = 0
                while (i < permissions.size) {
                    perms[permissions[i]] = grantResults[i]
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED) allGranted = false
                    i++
                }

                if (allGranted) {
                    // All Permissions Granted
                    init()
                } else {
                    // Permission Denied
                    Toast.makeText(
                        this@MainActivity,
                        "Some Permission is Denied",
                        Toast.LENGTH_SHORT
                    )
                        .show()
                    finish()
                }
            }
            else -> super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    }

    private fun allPermissionsGranted(): Boolean {
        for (permission in getRequiredPermissions()!!) {
            permission?.let { ContextCompat.checkSelfPermission(this, it) }
            if (permission?.let { ContextCompat.checkSelfPermission(this, it) }
                != PackageManager.PERMISSION_GRANTED
            ) {
                return false

            }
        }
        return true
    }

    private fun init() {
        try {
            mtcnnFaceDetector = FaceDetector.create(assets)
        } catch (e: Exception) {
            Log.d(TAG, "Exception initializing MTCNNModel!${e.stackTraceToString()}")
        }

        try {
            videoDetector = EmotionPyTorchVideoClassifier(applicationContext)
        } catch (e: java.lang.Exception) {
            Log.e(TAG, "Exception initializing feature extractor!", e)
        }

        try {
            clf = LinearSVC(applicationContext)
        } catch (e: java.lang.Exception) {
            Log.e(TAG, "Exception initializing classifier!", e)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == REQUEST_ACCESS_TYPE) {
            content = data?.data
            val intentNextStep = Intent(this, GalleryActivity::class.java)
            startActivity(intentNextStep)
        }
    }

    fun startGallery(view: View) {
        val intent = Intent()
        intent.type = "video/*"
        intent.action = Intent.ACTION_PICK
        startActivityForResult(intent, REQUEST_ACCESS_TYPE)
    }

    fun startLive(view: View) {
        val intent = Intent(this@MainActivity, CameraActivity::class.java)
        startActivity(intent)
    }
}