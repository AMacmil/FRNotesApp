package com.example.frnotesapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import java.lang.Float.max
import java.lang.Integer.min

class OverlayView(context: Context, attrs: AttributeSet): View(context, attrs) {
    private var faceBoundingBox: Rect? = null
    private val paint = Paint()

    init {
        // Style for the bounding box
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 5f
    }

    private fun setFaceBoundingBox(box: Rect) {
        faceBoundingBox = box
        invalidate() // Redraw the view
    }

    fun transformAndSetFaceBoundingBox(originalBox: Rect, cameraWidth: Int, cameraHeight: Int, viewWidth: Int, viewHeight: Int) {
        val scale = max(viewWidth.toFloat() / cameraWidth, viewHeight.toFloat() / cameraHeight)

        val offsetX = (viewWidth - cameraWidth * scale) / 2
        val offsetY = (viewHeight - cameraHeight * scale) / 2

        val mirroredLeft = cameraWidth - originalBox.right
        val mirroredRight = cameraWidth - originalBox.left

        // Introduce a width reduction factor, adjust this as needed
        val widthReduction = 0.6 // Example: Reduce width to 60% of its original
        val heightReduction = 0.8 // Example: Reduce width to 80% of its original

        val transformedBox = Rect(
            (mirroredLeft.toFloat() * scale + offsetX).toInt(),
            (originalBox.top.toFloat() * scale + offsetY).toInt(),
            ((mirroredRight.toFloat() * scale + offsetX) * widthReduction).toInt(),
            ((originalBox.bottom.toFloat() * scale + offsetY) * heightReduction).toInt()
        )

        setFaceBoundingBox(transformedBox)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Draw the bounding box
        faceBoundingBox?.let { canvas.drawRect(it, paint) }

        /*// Draw the target oval inside the bounding box
        faceBoundingBox?.let {
            val centerX = (it.left + it.right) / 2f
            val centerY = (it.top + it.bottom) / 2f
            val radius = min(it.width(), it.height()) / 2.5f
            canvas.drawOval(centerX - radius, centerY - radius, centerX + radius, centerY + radius, paint)
        }*/
    }
}
