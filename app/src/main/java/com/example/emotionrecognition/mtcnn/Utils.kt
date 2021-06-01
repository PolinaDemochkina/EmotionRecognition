package com.example.emotionrecognition.mtcnn

import java.util.*

object Utils {
    fun flipDiag(data: FloatArray, h: Int, w: Int, stride: Int) {
        val tmp = FloatArray(w * h * stride)
        for (i in 0 until w * h * stride) tmp[i] = data[i]
        for (y in 0 until h) for (x in 0 until w) {
            for (z in 0 until stride) data[(x * h + y) * stride + z] = tmp[(y * w + x) * stride + z]
        }
    }

    fun expand(src: FloatArray, dst: Array<Array<FloatArray>>) {
        var idx = 0
        for (y in dst.indices) for (element in dst[0]) for (c in dst[0][0].indices) element[c] =
            src[idx++]
    }

    fun expandProb(src: FloatArray, dst: Array<FloatArray>) {
        var idx = 0
        for (y in dst.indices) for (x in dst[0].indices) dst[y][x] = src[idx++ * 2 + 1]
    }

    fun updateBoxes(boxes: Vector<Box>): Vector<Box> {
        val b = Vector<Box>()
        for (i in boxes.indices) if (!boxes[i].deleted) b.addElement(boxes[i])
        return b
    }
}