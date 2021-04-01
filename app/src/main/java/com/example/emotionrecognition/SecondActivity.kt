package com.example.emotionrecognition

import android.animation.ObjectAnimator
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import androidx.core.animation.doOnEnd
import kotlinx.android.synthetic.main.activity_second.*

class SecondActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_second)

        Photo.setImageURI(MainActivity.image)
        ProgressBar.max = 100
    }

    fun analyze(view: View) {
        Analyze.visibility = View.GONE
        AnalyzeText.visibility = View.VISIBLE
        ProgressBar.visibility = View.VISIBLE

        val progress = ObjectAnimator.ofInt(ProgressBar, "progress", 100)
        progress.setDuration(3000)
        progress.doOnEnd {
            AnalyzeText.visibility = View.INVISIBLE
            ProgressBar.visibility = View.INVISIBLE
            Success.visibility = View.VISIBLE
            TopPanel.visibility = View.VISIBLE
            Back.visibility = View.VISIBLE
            Download.visibility = View.VISIBLE
        }
        progress.start()
    }

    fun back(view: View) {
        finish()
    }
}