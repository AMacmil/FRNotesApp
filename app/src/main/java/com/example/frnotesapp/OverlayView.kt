package com.example.frnotesapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import java.lang.Float.max
import java.lang.Float.min

// OverlayView class, inherits from View
// for drawing bounding box of detected face over the PreviewView
class OverlayView(context: Context, attrs: AttributeSet): View(context, attrs) {
    // var to store bounding box of detected face
    private var faceBoundingBox: Rect? = null
    // graphics paint object for drawing the bounding box
    private val paint = Paint()

    init {
        // style for the bounding box
        // TODO draw in green when match==true
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 5f
    }

    // set a new bounding box and trigger redraw
    private fun setFaceBoundingBox(box: Rect) {
        faceBoundingBox = box
        invalidate() // causes onDraw to be called to clear and redraw the new bBox
    }

    // transform and set bounding box based on view and camera dimensions
    // this is trial and error based and does not give the exact shape of the bounding box used in the model, but is good
    // for user experience
    fun transformAndSetFaceBoundingBox(originalBox: Rect, cameraWidth: Int, cameraHeight: Int, viewWidth: Int, viewHeight: Int) {
        // calc scale based on the larger dimension ratio (view/camera)
        val scale = max(viewWidth.toFloat() / cameraWidth, viewHeight.toFloat() / cameraHeight)

        // calc offsets for centering bounding box in the view
        val offsetX = (viewWidth - cameraWidth * scale) / 2
        val offsetY = (viewHeight - cameraHeight * scale) / 2

        // mirror the bounding box horizontally (camera feed uses front camera so is mirrored)
        val mirroredLeft = cameraWidth - originalBox.right
        val mirroredRight = cameraWidth - originalBox.left

        // scale width & height of bBox
        val scaledWidth = (mirroredRight - mirroredLeft) * scale
        val scaledHeight = (originalBox.bottom - originalBox.top) * scale

        // reduction factors for width and height - box was too large initially
        val widthReduction = 0.5f
        val heightReduction = 0.7f

        // apply above reductions
        val reducedWidth = scaledWidth * widthReduction
        val reducedHeight = scaledHeight * heightReduction

        // recalc right & bottom coords
        val transformedRight = mirroredLeft * scale + offsetX + reducedWidth
        val transformedBottom = originalBox.top * scale + offsetY + reducedHeight

        // new Rect object
        val transformedBox = Rect(
            (mirroredLeft * scale + offsetX).toInt(),
            (originalBox.top * scale + offsetY).toInt(),
            transformedRight.toInt(),
            transformedBottom.toInt()
        )
        // update the bounding box with the transformed coordinates
        setFaceBoundingBox(transformedBox)
    }

    // draw the bounding box
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // radius for rounded corners
        val cornerRadiusX = 200f
        val cornerRadiusY = 200f

        faceBoundingBox?.let {
            // conversion as drawRoundRect requires RectF
            val rectF = RectF(it)
            canvas.drawRoundRect(rectF, cornerRadiusX, cornerRadiusY, paint)
        }
    }
}
