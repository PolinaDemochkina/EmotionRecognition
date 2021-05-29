package com.example.emotionrecognition

object Constants {
    const val COUNT_OF_FRAMES_PER_INFERENCE = 4
    const val TARGET_VIDEO_SIZE = 224
    const val TARGET_IMAGE_SIZE = 600
    const val TARGET_FACE_SIZE = 224
    const val MODEL_INPUT_SIZE = 3 * TARGET_FACE_SIZE * TARGET_FACE_SIZE
    const val MIN_FACE_SIZE = 32
}