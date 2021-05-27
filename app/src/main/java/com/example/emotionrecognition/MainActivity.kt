package com.example.emotionrecognition

import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.emotionrecognition.mtcnn.MTCNNModel
import kotlinx.android.synthetic.main.activity_second.*
import java.util.*

class MainActivity : AppCompatActivity() {
    companion object {
        var content: Uri? = null
        val TAG = "MainActivity"
        var mtcnnFaceDetector: MTCNNModel? = null
        var imageDetector: EmotionPyTorchClassifier? = null
        var videoDetector: EmotionPyTorchVideoClassifier? = null
        var clf: LinearSVC? = null
        val minFaceSize = 32
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

    fun nextStep(view: View) {
        selectImageInAlbum()
    }

    private fun selectImageInAlbum() {
        val intent = Intent()
        intent.type = "image/* video/*"
        intent.action = Intent.ACTION_PICK
        startActivityForResult(intent, REQUEST_ACCESS_TYPE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == REQUEST_ACCESS_TYPE) {
            content = data?.data
            Log.d(TAG, "uri $content")
            val intentNextStep = Intent(this, SecondActivity::class.java)
            startActivity(intentNextStep)
        }
    }

    private fun allPermissionsGranted(): Boolean {
        for (permission in getRequiredPermissions()!!) {
            val status = permission?.let { ContextCompat.checkSelfPermission(this, it) }
            if (permission?.let { ContextCompat.checkSelfPermission(this, it) }
                != PackageManager.PERMISSION_GRANTED
            ) {
                return false

            }
        }
        return true
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
                // Check for ACCESS_FINE_LOCATION
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

    private fun init() {
        try {
            mtcnnFaceDetector = MTCNNModel.Companion.create(assets)
        } catch (e: Exception) {
            Log.d(TAG, "Exception initializing MTCNNModel!${e.stackTraceToString()}")
        }

        try {
            imageDetector = EmotionPyTorchClassifier(applicationContext)
        } catch (e: java.lang.Exception) {
            Log.e(TAG, "Exception initializing feature extractor!", e)
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
}