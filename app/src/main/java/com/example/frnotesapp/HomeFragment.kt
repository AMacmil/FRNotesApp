package com.example.frnotesapp

import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.Manifest
import android.graphics.Matrix
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.camera.core.ImageAnalysis
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.ByteArrayOutputStream
import kotlinx.coroutines.*
import kotlin.math.pow
import kotlin.math.sqrt
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import org.tensorflow.lite.Interpreter

// Fragment class for the home page/tab - handles camera preview and face detection
class HomeFragment : Fragment() {
    // ViewModel - for managing UI
    private lateinit var viewModel: NotesViewModel
    // PreviewView for displaying camera feed
    private lateinit var viewFinder: PreviewView
    // OverlayView for drawing bounding box
    private lateinit var overlayView : OverlayView

    // TensorFlow Lite interpreter for running the TensorFlow Lite models
    private lateinit var tfliteMobilenet: Interpreter
    private lateinit var tfliteMobileFaceNet: Interpreter
    private lateinit var tflitePretrainedMobileFaceNet: Interpreter
    private lateinit var tfliteFaceNet: Interpreter

    // vector to store features for verification reference
    private var storedFeatureVector: FloatArray? = null

    // required for camera permission
    private val cameraPermissionRequestCode = 1001

    // flags to control camera / model operations
    // volatile ensures updates to these vars are immediately visible to other threads / coroutines
    // detection / verification occurs in a coroutine
    @Volatile
    private var captureReferenceImage = false
    @Volatile
    private var stopCamera = false
    @Volatile
    private var isModelRunning = false

    // Bitmap for storing the cropped face for verification
    private lateinit var croppedFaceForVerification : Bitmap

    // lifecycle method for creating the view, calls on creation
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // initialize ViewModel
        viewModel = activity?.run {
            ViewModelProvider(this)[NotesViewModel::class.java]
        } ?: throw Exception("Invalid Activity")  // TODO - handle exceptions

        // inflate layout for this fragment
        return inflater.inflate(R.layout.home_fragment, container, false)
    }

    // lifecycle method called after the view is created
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        loadMobileFacenetModel(requireActivity())

        viewFinder = view.findViewById(R.id.viewFinder)
        overlayView = view.findViewById(R.id.overlay)

        val cameraVerificationButton = view.findViewById<Button>(R.id.cameraVerificationButton)
        cameraVerificationButton.setOnClickListener {
            stopCamera = false
            requestCameraPermission()
        }

        val captureReferenceButton = view.findViewById<Button>(R.id.captureReferenceButton)
        captureReferenceButton.setOnClickListener {
            stopCamera = false
            captureReferenceImage = true
            requestCameraPermission()
        }

        viewFinder.post {
            val previewWidth = viewFinder.width
            val previewHeight = viewFinder.height
            // Use these dimensions for calculations
            Log.d("DIMENSIONS", "$previewWidth")
            Log.d("DIMENSIONS", "$previewHeight")

        }
    }// end onViewCreated


    // load TensorFlow Lite model from assets
    private fun loadModelFile(activity: Activity, modelName: String): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // load the MobileFaceNet model trained in CoLab
    private fun loadMobileFacenetModel(activity: Activity) {
        val model = loadModelFile(activity, "output_model.tflite")
        tfliteMobileFaceNet = Interpreter(model)
    }

    //the following methods are for requesting permissions required for android so I can use the camera and probably storage later
    private fun requestCameraPermission() {
        Log.d("requestCameraPermission", "Requesting camera permission")
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.d("requestCameraPermission", "Camera permission not granted")
            if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
                Toast.makeText(context, "Camera permission is needed to show the camera preview.", Toast.LENGTH_SHORT).show()
            }
            requestPermissions(arrayOf(Manifest.permission.CAMERA), cameraPermissionRequestCode)
        } else {
            Log.d("requestCameraPermission", "Camera permission already granted")
            startCamera()
        }
    }// end requestCameraPermission

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        Log.d("onRequestPermissionsResult", "Permission result received")
        when (requestCode) {
            cameraPermissionRequestCode -> {
                // if request is cancelled the result arrays are empty
                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    Log.d("onRequestPermissionsResult", "Camera permission granted")
                    startCamera()
                } else {
                    Log.d("onRequestPermissionsResult", "Camera permission denied")
                    Toast.makeText(context, "Camera permission denied", Toast.LENGTH_SHORT).show()
                }
                return
            }
            // probs need storage later
        }
    }// end onRequestPermissionsResult

    // starts camera for detection / verification
    @OptIn(ExperimentalGetImage::class) private fun startCamera() {
        Log.d("startCamera", "Starting camera")
        // initialise camera and bind its lifecycle to lifecycle owner (this Fragment)
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // camera view setup
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            // front camera as default
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            // setup for skipping frames
            var frameSkipCounter = 0
            val frameSkipRate = 5 // skip every X frames

            var rotationDegrees = 0

            // image analyser for analyzing the camera feed, detecting faces, and running face recognition model
            val imageAnalysis = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(ContextCompat.getMainExecutor(requireContext()), ImageAnalysis.Analyzer { imageProxy ->
                        val cameraFeedWidth = imageProxy.width
                        val cameraFeedHeight = imageProxy.height

                        Log.d("DIMENSIONS", "$cameraFeedWidth")
                        Log.d("DIMENSIONS", "$cameraFeedHeight")

                        if (stopCamera) {
                            Log.e("STOP CAMERA", "stop camera is: $stopCamera")
                            imageProxy.close()
                            return@Analyzer
                        }
                        if (frameSkipCounter != 0 && (frameSkipCounter % frameSkipRate == 0 && !isModelRunning)) {
                            Log.e("imageAnalysis", "analysing image")
                            rotationDegrees = imageProxy.imageInfo.rotationDegrees
                            val mediaImage = imageProxy.image
                            Log.d("startCamera", "Image format: ${mediaImage?.format}")
                            if (mediaImage != null) {
                                Log.e("imageAnalysis", "checkpoint 2")
                                val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)
                                val detector = FaceDetection.getClient()
                                detector.process(inputImage)
                                    .addOnSuccessListener { faces ->
                                        if (faces.isNotEmpty()) {
                                            val boundingBox = faces[0].boundingBox
                                            overlayView.transformAndSetFaceBoundingBox(
                                                boundingBox,
                                                imageProxy.width,
                                                imageProxy.height,
                                                viewFinder.width,
                                                viewFinder.height
                                            )
                                            croppedFaceForVerification = cropBitmap(mediaImage, boundingBox, rotationDegrees)
                                            val preprocessedImage = preprocessImageMobileFacenet(croppedFaceForVerification, 112)
                                            runMobileFaceNetOnCameraImage(preprocessedImage)
                                        } else {
                                            isModelRunning = false
                                            Log.d("DETECTION", "No face detected")
                                        }
                                        Log.d("DETECTION", "onSuccessListener entered: detection attempt complete")
                                    }
                                    .addOnFailureListener { e ->
                                        // TODO - properly handle failure
                                        Log.e("DETECTION", "onFailureListener entered: ${e.message}")
                                    }
                                    .addOnCompleteListener {
                                        // close the imageProxy once complete
                                        imageProxy.close()
                                        Log.e("DETECTION", "onCompleteListener entered")
                                    }
                            } else {
                                // close imageProxy if mediaImage==null
                                imageProxy.close()
                                Log.e("imageAnalysis", "checkpoint 5")
                            }
                        } else {
                            // close the ImageProxy to avoid memory leaks
                            imageProxy.close()
                        }
                        // increment counter
                        frameSkipCounter = (frameSkipCounter + 1) % 10000

                        // log every 25 frames
                        if (frameSkipCounter % 25 == 0) {
                            Log.e("Counter", "FRAME: $frameSkipCounter")
                            Log.e("Counter", "Is Model Running?: $isModelRunning")
                        }
                    })
                }//end image analyser

            try {
                // unbind camera before rebinding
                cameraProvider.unbindAll()
                isModelRunning = false
                // bind use cases to camera
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
                Log.e("imageAnalysis", "checkpoint 6")
            } catch(exc: Exception) {
                Log.e("Camera", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(requireContext()))
    } // end startCamera

    private fun stopCamera() {
        val cameraProvider = ProcessCameraProvider.getInstance(requireContext()).get()
        cameraProvider.unbindAll()
    } // end stopCamera

    // runs face recognition on the captured image
    private fun runMobileFaceNetOnCameraImage(face: ByteBuffer): FloatArray {
        val outputSize = 128 // size for MobileFaceNet output

        // run inference on image
        val newImageOutput = Array(1) { FloatArray(outputSize) }
        tfliteMobileFaceNet.run(face, newImageOutput)
        val newImageFeatures = newImageOutput[0]

        // if capturing the reference image (indicated by captureReferenceImage) then store feature
        // vector for future and return so the rest of the function is skipped
        if (captureReferenceImage) {
            captureReferenceImage=false
            storedFeatureVector = newImageFeatures
            Log.e("CURRENT", "reference features stored")
            stopCamera = true
            stopCamera()
            return newImageFeatures
        }

        var comparisonVector = FloatArray(outputSize)

        if(storedFeatureVector != null){
            comparisonVector = storedFeatureVector as FloatArray
        }

        // compare feature vectors to determine a match based on cosine similarity
        // TODO base match on more than just cosine similarity?
        val similarity = calculateCosineSimilarity(newImageFeatures, comparisonVector)
        val isMatch = similarity > MATCH_THRESHOLD

        Log.d("VerificationTest", "Similarity: $similarity, Match: $isMatch")

        if (isMatch) {
            // show toast, stop camera, and set isAuthenticated to true
            Toast.makeText(context, "Match found!", Toast.LENGTH_SHORT).show()
            stopCamera = true
            stopCamera()
            (activity as? MainActivity)?.isAuthenticated = true
        } else {
            //Toast.makeText(context, "No match found.", Toast.LENGTH_SHORT).show()
            // TODO - decide what to do here
            //  - if I continue to do nothing it allows the user to keep trying after an initial failure
        }
        return newImageFeatures
    }// end runMobileFaceNetOnCameraImage

    // crops a given Image to the area defined by the bounding box.
    // converts Image to Bitmap via YUV
    private fun cropBitmap(image: Image, boundingBox: Rect, rotationDegrees: Int): Bitmap {
        Log.d("CROP", "cropBitmap begun")
        // extract Y and VU planes from the image to create an NV21 byte array
        val yBuffer = image.planes[0].buffer // Y
        val vuBuffer = image.planes[2].buffer // VU

        // calc size of Y and VU buffers
        val ySize = yBuffer.remaining()
        val vuSize = vuBuffer.remaining()

        // byte array to hold the NV21 data
        val nv21 = ByteArray(ySize + vuSize)

        // copy data from buffers into NV21 byte array
        yBuffer.get(nv21, 0, ySize)
        vuBuffer.get(nv21, ySize, vuSize)

        // create YuvImage object from NV21 data, width, and height of the original image
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)

        // convert YuvImage to jpeg -> bitmap
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // rotate the bitmap to match the orientation used during face detection
        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }
        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        // adjust bounding box to ensure it's within bitmap bounds
        val adjustedBox = Rect(
            boundingBox.left.coerceAtLeast(0),
            boundingBox.top.coerceAtLeast(0),
            boundingBox.right.coerceAtMost(rotatedBitmap.width),
            boundingBox.bottom.coerceAtMost(rotatedBitmap.height)
        )

        // crop bitmap to the adjusted bBox
        return Bitmap.createBitmap(
            rotatedBitmap,
            adjustedBox.left,
            adjustedBox.top,
            adjustedBox.width(),
            adjustedBox.height()
        )
    }// end cropBitmap

    // process image for use with the MobileFaceNet model
    private fun preprocessImageMobileFacenet(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        // resize image to match input size (112x112)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // create ByteBuffer to hold the image data
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        // normalize pixel values
        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[pixel++]
                // normalize and add RGB values to the ByteBuffer
                byteBuffer.putFloat(((pixelValue shr 16 and 0xFF) - 128) * 0.0078125f)  // Red
                byteBuffer.putFloat(((pixelValue shr 8 and 0xFF) - 128) * 0.0078125f)   // Green
                byteBuffer.putFloat(((pixelValue and 0xFF) - 128) * 0.0078125f)         // Blue
            }
        }
        return byteBuffer
    }// end preprocessImageMobileFacenet

    companion object {
        private const val MATCH_THRESHOLD = 0.7
        // TODO - fine tune this, maybe add more e.g. EUCLIDEAN_MATCH_THRESHOLD / COSINE_MATCH...
    }

    // calc Euclidean distance between two feature vectors
    private fun euclideanDistance(vec1: FloatArray, vec2: FloatArray): Float {
        var sum = 0.0
        for (i in vec1.indices) {
            // calc square of difference between elements of two vectors
            sum += (vec1[i] - vec2[i]).let { it * it }
        }
        // return square root of the sum (Euclidean distance)
        return sqrt(sum).toFloat()
    }// end euclideanDistance

    // calc standard deviation of a set of values
    private fun calculateStandardDeviation(values: FloatArray): Double {
        val mean = values.average()
        // calc sum of the squared differences from the mean
        var sum = 0.0
        for (value in values) {
            sum += (value - mean).pow(2)
        }
        // return square root of average of the squared differences (standard deviation)
        return sqrt(sum / values.size)
    }// end calculateStandardDeviation

    // calc cosine similarity between two feature vectors
    private fun calculateCosineSimilarity(vec1: FloatArray, vec2: FloatArray): Double {
        var dotProduct = 0.0
        var normVec1 = 0.0
        var normVec2 = 0.0
        for (i in vec1.indices) {
            // accumulate the dot product of vectors and the squares of their norms
            dotProduct += vec1[i] * vec2[i]
            normVec1 += vec1[i].pow(2)
            normVec2 += vec2[i].pow(2)
        }
        // return dot product divided by the product of the vectors norms (cosine similarity)
        return dotProduct / (sqrt(normVec1) * sqrt(normVec2))
    }// end calculateCosineSimilarity

}// end class HomeFragment