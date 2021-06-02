package com.example.emotionrecognition

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.gson.Gson
import java.io.*
import java.util.*


class LinearSVC(context: Context) {
    companion object {
        @Throws(IOException::class)
        private fun getFileFromAsset(context: Context): File? {
            val assetManager: AssetManager = context.assets
            var inputStream: InputStream? = null
            var outputStream: OutputStream? = null
            var file: File? = null
            val filename = "data.json"
            try {
                inputStream = assetManager.open(filename)
                file = File(context.cacheDir, filename)
                outputStream = FileOutputStream(file)
                val buffer = ByteArray(4 * 1024) // or other buffer size
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            } catch (e: Exception) {
                e.printStackTrace() // handle exception, define IOException and others
            } finally {
                try {
                    outputStream?.close()
                    inputStream?.close()
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
            return file
        }
    }

    internal class Classifier {
        val coefficients: Array<DoubleArray> = arrayOf()
        val intercepts: DoubleArray = doubleArrayOf()
    }

    private val clf: Classifier

    fun predict(features: List<Float>): Int {
        var classIdx = 0
        var classVal = Double.NEGATIVE_INFINITY
        var i = 0
        val il = clf.intercepts.size
        while (i < il) {
            var prob = 0.0
            var j = 0
            val jl: Int = clf.coefficients[0].size
            while (j < jl) {
                prob += clf.coefficients[i][j] * features[j]
                j++
            }
            if (prob + clf.intercepts[i] > classVal) {
                classVal = prob + clf.intercepts[i]
                classIdx = i
            }
            i++
        }
        return classIdx
    }

    init {
        val jsonStr = Scanner(getFileFromAsset(context)).useDelimiter("\\Z").next()
        Log.e(MainActivity.TAG, jsonStr)
        clf = Gson().fromJson(jsonStr, Classifier::class.java)
    }
}