package com.example.emotionrecognition

object Constants {
    const val COUNT_OF_FRAMES_PER_INFERENCE = 4
    const val TARGET_VIDEO_SIZE = 224
    const val MODEL_INPUT_SIZE =
        COUNT_OF_FRAMES_PER_INFERENCE * 3 * TARGET_VIDEO_SIZE * TARGET_VIDEO_SIZE
}