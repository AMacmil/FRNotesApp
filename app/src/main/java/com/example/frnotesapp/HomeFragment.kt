package com.example.frnotesapp

// Importing necessary Android and TensorFlow classes
import android.app.Activity
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import android.widget.Button
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.pow
import kotlin.math.sqrt
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.*
import kotlin.coroutines.resume
import android.widget.Toast
import kotlin.math.abs

// Fragment class for the first page/tab
class HomeFragment : Fragment() {
    // declare ViewModel - for managing UI-related data in a lifecycle-conscious way
    private lateinit var viewModel: NotesViewModel

    // TensorFlow Lite interpreter for running the TensorFlow Lite models
    private lateinit var tfliteMobilenet: Interpreter
    private lateinit var tfliteMobileFaceNet: Interpreter
    private lateinit var tfliteFaceNet: Interpreter

    // Function to load the TensorFlow Lite model from the assets
    private fun loadModelFile(activity: Activity, modelName: String): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Function to load the Mobilenet Feature Vector model
    private fun loadMobilenetFVModel(activity: Activity) {
        val model = loadModelFile(activity, "mobilenet_v3_feature_vector.tflite")
        tfliteMobilenet = Interpreter(model)
    }

    // Function to load the FaceNet model
    private fun loadFacenetModel(activity: Activity) {
        val model = loadModelFile(activity, "facenet.tflite")
        tfliteFaceNet = Interpreter(model)
    }

    // Function to load the FaceNet model
    private fun loadMobileFacenetModel(activity: Activity) {
        val model = loadModelFile(activity, "pretrained_mfnet.tflite")
        tfliteMobileFaceNet = Interpreter(model)
    }

    // Function to preprocess the image before feeding it to the Mobilenet FV model
    private fun preprocessImageMobilenetFV(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        // Decode the image file and resize it to the expected input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Create a ByteBuffer to hold the image data
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Normalize pixel values and add them to the ByteBuffer
        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[pixel++]
                byteBuffer.putFloat(((pixelValue shr 16 and 0xFF) / 255f) - 0.5f) // Red
                byteBuffer.putFloat(((pixelValue shr 8 and 0xFF) / 255f) - 0.5f)  // Green
                byteBuffer.putFloat(((pixelValue and 0xFF) / 255f) - 0.5f)       // Blue
            }
        }

        return byteBuffer
    }

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
        loadMobilenetFVModel(requireActivity())
        //loadFacenetModel(requireActivity()) // I couldn't get this to work
        loadMobileFacenetModel(requireActivity())

        // Loading images as bitmaps - TODO these are for testing and need replaced with uploaded images
        val bitmap1 = BitmapFactory.decodeResource(resources, R.drawable.face_1)
        val bitmap2 = BitmapFactory.decodeResource(resources, R.drawable.face_2)
        val bitmap3 = BitmapFactory.decodeResource(resources, R.drawable.face_3)
        val bitmap4 = BitmapFactory.decodeResource(resources, R.drawable.face_4)
        val bitmap5 = BitmapFactory.decodeResource(resources, R.drawable.face_5)
        val bitmapA = BitmapFactory.decodeResource(resources, R.drawable.face_a)
        val bitmapB = BitmapFactory.decodeResource(resources, R.drawable.face_b)
        val bitmapC = BitmapFactory.decodeResource(resources, R.drawable.face_c)
        val bitmapD = BitmapFactory.decodeResource(resources, R.drawable.face_d)

        val preprocessMobilenetFVButton = view.findViewById<Button>(R.id.preprocessMobilenetFVButton)
        preprocessMobilenetFVButton.setOnClickListener {
            Log.d("HomeFragment", "preprocessMobilenetFVButton button clicked")

            // preprocessing images
            Log.d("HomeFragment", "starting preprocessing")
            val inputSize = 224 // Setting inputSize to the required size for MobileNet
            val preprocessedImage1 = preprocessImageMobilenetFV(bitmap1, inputSize)
            val preprocessedImage2 = preprocessImageMobilenetFV(bitmap2, inputSize)
            val preprocessedImage3 = preprocessImageMobilenetFV(bitmap3, inputSize)
            val preprocessedImage4 = preprocessImageMobilenetFV(bitmap4, inputSize)
            val preprocessedImage5 = preprocessImageMobilenetFV(bitmap5, inputSize)
            val preprocessedImageA = preprocessImageMobilenetFV(bitmapA, inputSize)
            val preprocessedImageB = preprocessImageMobilenetFV(bitmapB, inputSize)
            val preprocessedImageC = preprocessImageMobilenetFV(bitmapC, inputSize)
            val preprocessedImageD = preprocessImageMobilenetFV(bitmapD, inputSize)
            Log.d("HomeFragment", "Preprocessing done, image ready for model")

            // Declaring outputs
            val output1 = Array(1) { FloatArray(1280) }
            val output2 = Array(1) { FloatArray(1280) }
            val output3 = Array(1) { FloatArray(1280) }
            val output4 = Array(1) { FloatArray(1280) }
            val output5 = Array(1) { FloatArray(1280) }
            val outputA = Array(1) { FloatArray(1280) }
            val outputB = Array(1) { FloatArray(1280) }
            val outputC = Array(1) { FloatArray(1280) }
            val outputD = Array(1) { FloatArray(1280) }

            // Running inference on the preprocessed images
            Log.d("HomeFragment", "Running inference")
            tfliteMobilenet.run(preprocessedImage1, output1)
            tfliteMobilenet.run(preprocessedImage2, output2)
            tfliteMobilenet.run(preprocessedImage3, output3)
            tfliteMobilenet.run(preprocessedImage4, output4)
            tfliteMobilenet.run(preprocessedImage5, output5)
            tfliteMobilenet.run(preprocessedImageA, outputA)
            tfliteMobilenet.run(preprocessedImageB, outputB)
            tfliteMobilenet.run(preprocessedImageC, outputC)
            tfliteMobilenet.run(preprocessedImageD, outputD)
            Log.d("HomeFragment", "Inference done")

            val featureVector1 = output1[0]
            val featureVector2 = output2[0]
            val featureVector3 = output3[0]
            val featureVector4 = output4[0]
            val featureVector5 = output5[0]
            val featureVectorA = outputA[0]
            val featureVectorB = outputB[0]
            val featureVectorC = outputC[0]
            val featureVectorD = outputD[0]

            // Logging feature vectors and comparing them for similarity
            Log.d("HomeFragment", "Feature Vector1: ${featureVector1.joinToString(", ")}")
            Log.d("HomeFragment", "Feature Vector2: ${featureVector2.joinToString(", ")}")
            Log.d("HomeFragment", "Feature Vector3: ${featureVector3.joinToString(", ")}")
            Log.d("HomeFragment", "Feature Vector4: ${featureVector4.joinToString(", ")}")
            Log.d("HomeFragment", "Feature Vector5: ${featureVector5.joinToString(", ")}")
            Log.d("HomeFragment", "Feature Vector A: ${featureVectorA.joinToString(", ")}")
            Log.d("HomeFragment", "Feature Vector B: ${featureVectorB.joinToString(", ")}")
            Log.d("HomeFragment", "Feature Vector C: ${featureVectorC.joinToString(", ")}")
            Log.d("HomeFragment", "Feature Vector D: ${featureVectorD.joinToString(", ")}")

            /*// Basic Statistical Analysis
            val mean1 = featureVector1.average()
            val max1 = featureVector1.maxOrNull()
            val min1 = featureVector1.minOrNull()
            val std1 = calculateStandardDeviation(featureVector1)

            val mean2 = featureVector2.average()
            val max2 = featureVector2.maxOrNull()
            val min2 = featureVector2.minOrNull()
            val std2 = calculateStandardDeviation(featureVector2)

            val mean3 = featureVector3.average()
            val max3 = featureVector3.maxOrNull()
            val min3 = featureVector3.minOrNull()
            val std3 = calculateStandardDeviation(featureVector3)

            val mean4 = featureVector4.average()
            val max4 = featureVector4.maxOrNull()
            val min4 = featureVector4.minOrNull()
            val std4 = calculateStandardDeviation(featureVector4)

            val mean5 = featureVector5.average()
            val max5 = featureVector5.maxOrNull()
            val min5 = featureVector5.minOrNull()
            val std5 = calculateStandardDeviation(featureVector5)

            val meanA = featureVectorA.average()
            val maxA = featureVectorA.maxOrNull()
            val minA = featureVectorA.minOrNull()
            val stdA = calculateStandardDeviation(featureVectorA)

            val meanB = featureVectorB.average()
            val maxB = featureVectorB.maxOrNull()
            val minB = featureVectorB.minOrNull()
            val stdB = calculateStandardDeviation(featureVectorB)

            val meanC = featureVectorC.average()
            val maxC = featureVectorC.maxOrNull()
            val minC = featureVectorC.minOrNull()
            val stdC = calculateStandardDeviation(featureVectorC)

            val meanD = featureVectorD.average()
            val maxD = featureVectorD.maxOrNull()
            val minD = featureVectorD.minOrNull()
            val stdD = calculateStandardDeviation(featureVectorD)
*/
            /*Log.d("HomeFragment", "Mean 1: $mean1, Max: $max1, Min: $min1, Std: $std1")
            Log.d("HomeFragment", "Mean 2: $mean2, Max: $max2, Min: $min2, Std: $std2")
            Log.d("HomeFragment", "Mean 3: $mean3, Max: $max3, Min: $min3, Std: $std3")
            Log.d("HomeFragment", "Mean 4: $mean4, Max: $max4, Min: $min4, Std: $std4")
            Log.d("HomeFragment", "Mean 5: $mean5, Max: $max5, Min: $min5, Std: $std5")
            Log.d("HomeFragment", "Mean A: $meanA, Max: $maxA, Min: $minA, Std: $stdA")
            Log.d("HomeFragment", "Mean B: $meanB, Max: $maxB, Min: $minB, Std: $stdB")
            Log.d("HomeFragment", "Mean C: $meanC, Max: $maxC, Min: $minC, Std: $stdC")
            Log.d("HomeFragment", "Mean D: $meanD, Max: $maxD, Min: $minD, Std: $stdD")*/

            // Calculating and logging cosine similarity and Euclidean distance between feature vectors
            // Using the function to compare the two feature vectors
            val cosineSimilarity1 = calculateCosineSimilarity(featureVector1, featureVector2)
            val cosineSimilarity2 = calculateCosineSimilarity(featureVector1, featureVector3)
            val cosineSimilarity3 = calculateCosineSimilarity(featureVector1, featureVector4)
            val cosineSimilarity4 = calculateCosineSimilarity(featureVector1, featureVector5)
            val cosineSimilarityA = calculateCosineSimilarity(featureVector1, featureVectorA)
            val cosineSimilarityB = calculateCosineSimilarity(featureVector1, featureVectorB)
            val cosineSimilarityC = calculateCosineSimilarity(featureVector1, featureVectorC)
            val cosineSimilarityD = calculateCosineSimilarity(featureVector1, featureVectorD)

            val distance1 = euclideanDistance(featureVector1, featureVector2)
            val distance2 = euclideanDistance(featureVector1, featureVector3)
            val distance3 = euclideanDistance(featureVector1, featureVector4)
            val distanceA = euclideanDistance(featureVector1, featureVectorA)
            val distanceB = euclideanDistance(featureVector1, featureVectorB)
            val distanceC = euclideanDistance(featureVector1, featureVectorC)
            val distanceD = euclideanDistance(featureVector1, featureVectorD)

            Log.d("Cosine Similarity", "A cosine similarity close to 1 indicates a very small angle, suggesting high similarity (or near-identical orientation) between the vectors. A value close to -1 indicates a large angle, signifying dissimilarity.")
            Log.d("Cosine Similarity", "Similarity between 2 pics of me: $cosineSimilarity1")
            Log.d("Cosine Similarity", "Similarity between 2 pics of me: $cosineSimilarity2")
            Log.d("Cosine Similarity", "Similarity between 2 pics of me: $cosineSimilarity3")
            Log.d("Cosine Similarity", "Similarity between 2 pics of me: $cosineSimilarity4")
            Log.d("Cosine Similarity", "Similarity between me and a middle aged asian woman: $cosineSimilarityA")
            Log.d("Cosine Similarity", "Similarity between me and a young black woman: $cosineSimilarityB")
            Log.d("Cosine Similarity", "Similarity between me and a young white woman: $cosineSimilarityC")
            Log.d("Cosine Similarity", "Similarity between me and a guy who looks a bit like me: $cosineSimilarityD")
            Log.d("Euclidean Distance", "The closer this value is to 0, the more similar the feature vectors are, indicating a higher likelihood that the images are of the same person")
            Log.d("Euclidean Distance", "Distance between 2 pics of me: $distance1")
            Log.d("Euclidean Distance", "Distance between 2 pics of me: $distance2")
            Log.d("Euclidean Distance", "Distance between 2 pics of me: $distance3")
            Log.d("Euclidean Distance", "Distance between me and a middle aged asian woman: $distanceA")
            Log.d("Euclidean Distance", "Distance between me and a young black woman: $distanceB")
            Log.d("Euclidean Distance", "Distance between me and a young white woman: $distanceC")
            Log.d("Euclidean Distance", "Distance between me and a guy who looks a bit like me: $distanceD")

        }// end clickListener for preprocessMobilenetFVButton

        val preprocessMobileFacenetButton = view.findViewById<Button>(R.id.preprocessMobileFacenetButton)
        preprocessMobileFacenetButton.setOnClickListener {
            CoroutineScope(Dispatchers.Main).launch {

                val bitmaps = listOf(
                    bitmap1,
                    bitmap2,
                    bitmap3,
                    bitmap4,
                    bitmap5,
                    bitmapA,
                    bitmapB,
                    bitmapC,
                    bitmapD
                )

                // Run cropping and preprocessing in a coroutine
                Log.d("PREPROCESSING", "Start Preprocessing")
                val preprocessedImages = withContext(Dispatchers.IO) {
                    cropAndPreprocessImages(bitmaps)
                }
                Log.d("PREPROCESSING", "End Preprocessing")

                // Continue with the inference process using preprocessed images
                Log.d("INFERENCE", "Start Inference")
                runInference(preprocessedImages)
                Log.d("INFERENCE", "End Inference")
            }// end coroutine scope
        }// end clickListener for preprocessMobileFacenetButton

        val verificationTestButton = view.findViewById<Button>(R.id.verificationTestButton)
        verificationTestButton.setOnClickListener {
            performVerificationTest()
        }
    }// end onViewCreated

    private fun performVerificationTest() {
        CoroutineScope(Dispatchers.Main).launch {
            // Load or capture a new image for testing
            val newTestImageBitmap = captureOrLoadNewImage()

            // Preprocess the new test image
            val preprocessedTestImage = withContext(Dispatchers.IO) {
                cropAndPreprocessImage(newTestImageBitmap, 112)
            }

            // Load the pre-stored reference image
            val referenceImageBitmap = loadReferenceImage()
            val preprocessedReferenceImage = withContext(Dispatchers.IO) {
                cropAndPreprocessImage(referenceImageBitmap, 112)
            }

            // Perform the verification test
            verifyNewImageAgainstReference(preprocessedTestImage, preprocessedReferenceImage)
        }
    }

    private fun captureOrLoadNewImage(): Bitmap {
        // TODO - replace with camera / upload function
        return BitmapFactory.decodeResource(resources, R.drawable.face_2)
    }

    private fun loadReferenceImage(): Bitmap {
        // TODO - replace with image stored from upload
        return BitmapFactory.decodeResource(resources, R.drawable.face_1)
    }

    private fun verifyNewImageAgainstReference(newImage: ByteBuffer, referenceImage: ByteBuffer) {
        val outputSize = 128 // Adjust this according to your model's output

        // Run model inference on new image
        val newImageOutput = Array(1) { FloatArray(outputSize) }
        tfliteMobileFaceNet.run(newImage, newImageOutput)
        val newImageFeatures = newImageOutput[0]

        // Run model inference on reference image
        val referenceImageOutput = Array(1) { FloatArray(outputSize) }
        tfliteMobileFaceNet.run(referenceImage, referenceImageOutput)
        val referenceImageFeatures = referenceImageOutput[0]

        // Compare feature vectors to determine a match
        val similarity = calculateCosineSimilarity(newImageFeatures, referenceImageFeatures)
        val isMatch = similarity > MATCH_THRESHOLD

        // Log the results for debugging
        Log.d("VerificationTest", "Similarity: $similarity, Match: $isMatch")

        // Show a toast message if a match is found
        if (isMatch) {
            Toast.makeText(context, "Match found!", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(context, "No match found.", Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        private const val MATCH_THRESHOLD = 0.7 // TODO - fine tune this
    }


    private suspend fun cropAndPreprocessImages(bitmaps: List<Bitmap>): List<ByteBuffer> {
        val inputSize = 112 // Required size for MobileNet
        return bitmaps.map { bitmap ->
            withContext(Dispatchers.Default) {
                cropAndPreprocessImage(bitmap, inputSize)
            }
        }
    }

    private suspend fun cropAndPreprocessImage(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        var preprocessedImage = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        suspendCancellableCoroutine<Unit> { continuation ->
            cropForMobileFacenet(bitmap) { croppedFaceBitmap ->
                croppedFaceBitmap?.let {
                    Log.d("Cropping", "Cropping Complete")
                    preprocessedImage = preprocessImageMobileFacenet(croppedFaceBitmap, inputSize)
                } ?: run {
                    Log.d("Cropping", "Cropping Failed")
                    preprocessedImage = preprocessImageMobileFacenet(bitmap, inputSize)
                }
                continuation.resume(Unit)
            }
        }
        return preprocessedImage
    }

    /*private fun runInference(preprocessedImages: List<ByteBuffer>) {
        val outputSize = 128 // Set this based on your model's output

        val outputs = preprocessedImages.map { imageBuffer ->
            val output = Array(1) { FloatArray(outputSize) }
            tfliteMobileFaceNet.run(imageBuffer, output)
            output[0]
        }

        // Process the outputs
        for (output in outputs) {
            // Log or process the feature vectors
            Log.d("HomeFragment", "Feature Vector: ${output.joinToString(", ")}")
        }

        // Example of calculating cosine similarity between two feature vectors
        val cosineSimilarity = calculateCosineSimilarity(outputs[0], outputs[1])
        Log.d("Cosine Similarity", "Similarity between two feature vectors: $cosineSimilarity")

        // Example of calculating Euclidean distance
        val distance = euclideanDistance(outputs[0], outputs[1])
        Log.d("Euclidean Distance", "Distance between two feature vectors: $distance")
    }*/

    private fun runInference(preprocessedImages: List<ByteBuffer>) {
        val outputSize = 128 // Set this based on your model's output

        val outputs = preprocessedImages.map { imageBuffer ->
            val output = Array(1) { FloatArray(outputSize) }
            tfliteMobileFaceNet.run(imageBuffer, output)
            output[0]
        }

        // Comparing feature vectors of the same person (images 1-5)
        for (i in 0 until 4) { // Compare each image with the next one
            val cosineSimilarity = calculateCosineSimilarity(outputs[i], outputs[i + 1])
            val distance = euclideanDistance(outputs[i], outputs[i + 1])
            Log.d("Comparison", "Same person: Images ${i + 1} and ${i + 2}, Cosine Similarity: $cosineSimilarity, Euclidean Distance: $distance")
        }

        // Comparing feature vectors of different people (images 1 with A-D)
        for (i in 5 until outputs.size) {
            val cosineSimilarity = calculateCosineSimilarity(outputs[0], outputs[i])
            val distance = euclideanDistance(outputs[0], outputs[i])
            Log.d("Comparison", "Different people: Image 1 and ${'A' + i - 5}, Cosine Similarity: $cosineSimilarity, Euclidean Distance: $distance")
        }
    }



    // Function to use ML Kit to detect and crop the face
    private fun cropForMobileFacenet(bitmap: Bitmap, completion: (Bitmap?) -> Unit) {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()

        val detector = FaceDetection.getClient(options)
        val image = InputImage.fromBitmap(bitmap, 0)

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    // For simplicity, we're using the first detected face.
                    val face = faces.first()
                    val bounds = face.boundingBox

                    // Crop the face from the bitmap
                    val faceBitmap = Bitmap.createBitmap(
                        bitmap,
                        bounds.left.coerceAtLeast(0),
                        bounds.top.coerceAtLeast(0),
                        bounds.width().coerceAtMost(bitmap.width - bounds.left),
                        bounds.height().coerceAtMost(bitmap.height - bounds.top)
                    )
                    completion(faceBitmap)
                } else {
                    // No faces detected, return null or the original bitmap
                    completion(null)
                }
            }
            .addOnFailureListener { e ->
                // Handle the error, return null or the original bitmap
                completion(null)
            }
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