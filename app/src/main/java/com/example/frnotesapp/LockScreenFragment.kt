package com.example.frnotesapp

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.pow
import kotlin.math.sqrt

class LockScreenFragment : Fragment() {
    // declare ViewModel - for managing UI-related data in a lifecycle-conscious way
    private lateinit var viewModel: NotesViewModel
    // PreviewView for displaying camera feed
    private lateinit var viewFinder: PreviewView
    // TensorFlow Lite interpreter for running the TensorFlow Lite models
    private lateinit var tfliteMobilenet: Interpreter
    private lateinit var tfliteMobileFaceNet: Interpreter
    private lateinit var tflitePretrainedMobileFaceNet: Interpreter
    private lateinit var tfliteFaceNet: Interpreter
    // Vector to store features stored for verification reference
    private var storedFeatureVector: FloatArray? = null
    private val cameraPermissionRequestCode = 1001
    @Volatile
    private var captureReferenceImage = false
    @Volatile
    private var stopCamera = false
    @Volatile
    private var isModelRunning = false
    private lateinit var croppedFaceForVerification : Bitmap

    // Lifecycle method for creating the view
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // initialize ViewModel
        viewModel = activity?.run {
            ViewModelProvider(this)[NotesViewModel::class.java]
        } ?: throw Exception("Invalid Activity")  // throw an exception if the activity is invalid

        // inflate layout for this fragment
        return inflater.inflate(R.layout.home_fragment, container, false)
    }

    // Lifecycle method called after the view is created
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        loadMobileFacenetModel(requireActivity())

        viewFinder = view.findViewById(R.id.viewFinder)

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
    }// end onViewCreated


    // Function to load the TensorFlow Lite model from the assets
    private fun loadModelFile(activity: Activity, modelName: String): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Function to load the MobileFaceNet model trained in CoLab
    private fun loadMobileFacenetModel(activity: Activity) {
        val model = loadModelFile(activity, "output_model.tflite")
        tfliteMobileFaceNet = Interpreter(model)
    }

    //The following methods are for requesting permissions required for android so I can use the camera and probably storage later
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
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        Log.d("onRequestPermissionsResult", "Permission result received")
        when (requestCode) {
            cameraPermissionRequestCode -> {
                // If request is cancelled, the result arrays are empty.
                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    Log.d("onRequestPermissionsResult", "Camera permission granted")
                    startCamera()
                } else {
                    Log.d("onRequestPermissionsResult", "Camera permission denied")
                    Toast.makeText(context, "Camera permission denied", Toast.LENGTH_SHORT).show()
                }
                return
            }
            // Other permission requests and results - probs need storage later
        }
    }

    @OptIn(ExperimentalGetImage::class) private fun startCamera() {
        Log.d("startCamera", "Starting camera")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            var frameSkipCounter = 0
            val frameSkipRate = 50 // Skip every X frames
            var rotationDegrees = 0
            val imageAnalysis = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(ContextCompat.getMainExecutor(requireContext()), ImageAnalysis.Analyzer { imageProxy ->
                        if (stopCamera) {
                            Log.e("STOPCAMERA", "stop camera is: $stopCamera")
                            imageProxy.close()
                            return@Analyzer
                        }
                        if (frameSkipCounter != 0 && (frameSkipCounter % frameSkipRate == 0 && !isModelRunning)) {
                            //isModelRunning = true
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
                                        // Handle any errors during ML Kit face detection
                                        Log.e("DETECTION", "onFailureListener entered: ${e.message}")
                                    }
                                    .addOnCompleteListener {
                                        // After done with the frame, you must close the imageProxy
                                        imageProxy.close()
                                        Log.e("DETECTION", "onCompleteListener entered")
                                    }
                            } else {
                                // Close the imageProxy if mediaImage is null
                                imageProxy.close()
                                Log.e("imageAnalysis", "checkpoint 5")
                            }
                        } else {
                            // Close the ImageProxy to avoid memory leaks
                            imageProxy.close()
                        }
                        frameSkipCounter = (frameSkipCounter + 1) % 10000
                        // Log every 25 frames
                        if (frameSkipCounter % 25 == 0) {
                            Log.e("Counter", "FRAME: $frameSkipCounter")
                            Log.e("Counter", "Is Model Running?: $isModelRunning")
                        }
                    })
                }

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()
                isModelRunning = false
                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis)
                //this, cameraSelector, preview)
                Log.e("imageAnalysis", "checkpoint 6")
            } catch(exc: Exception) {
                Log.e("Camera", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(requireContext()))
    } // end startCamera

    private fun stopCamera() {
        val cameraProvider = ProcessCameraProvider.getInstance(requireContext()).get()
        cameraProvider.unbindAll()
    }

    private fun runMobileFaceNetOnCameraImage(face: ByteBuffer): FloatArray {
        val outputSize = 128 // Adjust this according to your model's output

        // Run model inference on new image
        val newImageOutput = Array(1) { FloatArray(outputSize) }
        tfliteMobileFaceNet.run(face, newImageOutput)
        val newImageFeatures = newImageOutput[0]

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
        // Compare feature vectors to determine a match
        val similarity = calculateCosineSimilarity(newImageFeatures, comparisonVector)
        val isMatch = similarity > MATCH_THRESHOLD

        // Log the results for debugging
        Log.d("VerificationTest", "Similarity: $similarity, Match: $isMatch")

        // Show a toast message if a match is found
        if (isMatch) {
            Toast.makeText(context, "Match found!", Toast.LENGTH_SHORT).show()
            (activity as? MainActivity)?.isAuthenticated = true
        } else {
            Toast.makeText(context, "No match found.", Toast.LENGTH_SHORT).show()
        }
        return newImageFeatures
    }

    private fun cropBitmap(image: Image, boundingBox: Rect, rotationDegrees: Int): Bitmap {
        Log.d("CROP", "cropBitmap begun")
        val yBuffer = image.planes[0].buffer // Y
        val vuBuffer = image.planes[2].buffer // VU

        val ySize = yBuffer.remaining()
        val vuSize = vuBuffer.remaining()

        val nv21 = ByteArray(ySize + vuSize)

        yBuffer.get(nv21, 0, ySize)
        vuBuffer.get(nv21, ySize, vuSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)

        val imageBytes = out.toByteArray()
        var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // Rotate the bitmap to match the orientation used during face detection
        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }
        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        // Adjust the bounding box to ensure it's within the bitmap's bounds
        val adjustedBox = Rect(
            boundingBox.left.coerceAtLeast(0),
            boundingBox.top.coerceAtLeast(0),
            boundingBox.right.coerceAtMost(rotatedBitmap.width),
            boundingBox.bottom.coerceAtMost(rotatedBitmap.height)
        )

        // Crop the bitmap to the adjusted bounding box
        return Bitmap.createBitmap(
            rotatedBitmap,
            adjustedBox.left,
            adjustedBox.top,
            adjustedBox.width(),
            adjustedBox.height()
        )
    }

    private fun preprocessImageMobileFacenet(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        // Resize the image to match the input size (112x112)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Create a ByteBuffer to hold the image data
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Normalize pixel values
        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[pixel++]
                byteBuffer.putFloat(((pixelValue shr 16 and 0xFF) - 128) * 0.0078125f) // Red
                byteBuffer.putFloat(((pixelValue shr 8 and 0xFF) - 128) * 0.0078125f)  // Green
                byteBuffer.putFloat(((pixelValue and 0xFF) - 128) * 0.0078125f)       // Blue
            }
        }

        return byteBuffer
    }

    companion object {
        private const val MATCH_THRESHOLD = 0.7 // TODO - fine tune this
    }

    // Function to calculate the Euclidean distance between two feature vectors
    private fun euclideanDistance(vec1: FloatArray, vec2: FloatArray): Float {
        var sum = 0.0
        for (i in vec1.indices) {
            sum += (vec1[i] - vec2[i]).let { it * it }
        }
        return sqrt(sum).toFloat()
    }

    // Function to calculate the standard deviation of a set of values
    private fun calculateStandardDeviation(values: FloatArray): Double {
        val mean = values.average()
        var sum = 0.0
        for (value in values) {
            sum += (value - mean).pow(2)
        }
        return sqrt(sum / values.size)
    }

    // Function to calculate the cosine similarity between two feature vectors
    private fun calculateCosineSimilarity(vec1: FloatArray, vec2: FloatArray): Double {
        var dotProduct = 0.0
        var normVec1 = 0.0
        var normVec2 = 0.0
        for (i in vec1.indices) {
            dotProduct += vec1[i] * vec2[i]
            normVec1 += vec1[i].pow(2)
            normVec2 += vec2[i].pow(2)
        }
        return dotProduct / (sqrt(normVec1) * sqrt(normVec2))
    }
}