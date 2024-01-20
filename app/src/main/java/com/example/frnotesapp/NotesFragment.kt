package com.example.frnotesapp

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CalendarView
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

private const val REQUEST_RECORD_AUDIO_PERMISSION = 200  // permission request code for recording

// Fragment class for the second page/tab
class NotesFragment : Fragment() {
    // lateinit variables for UI components and state
    private lateinit var selectedImagePath: String
    private lateinit var pickMedia: ActivityResultLauncher<PickVisualMediaRequest>
    private lateinit var recordButton: Button
    private lateinit var stopRecordButton: Button
    private lateinit var cancelRecordButton: Button
    private lateinit var selectButton: Button
    private lateinit var uploadButton: Button
    private lateinit var selectedImageView: ImageView
    private lateinit var fileNameEditText: EditText
    private lateinit var imageTitleEditText: EditText
    private lateinit var v: View
    private lateinit var viewModel: NotesViewModel
    private lateinit var recorder: MediaRecorder
    // variables for permission to record audio
    private var permissionToRecordAccepted = false
    private var permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        viewModel = activity?.run {
            ViewModelProvider(this)[NotesViewModel::class.java]
        } ?: throw Exception("Invalid Activity")

        v = inflater.inflate(R.layout.notes_fragment, container, false)

        // pickMedia uses Android PhotoPicker to select image from gallery - returns result uri
        // and sets the image in the preview frame selectedImageView
        pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            if (uri != null) {
                val filePath = copyFileToInternalStorage(uri, requireContext().filesDir.path, requireContext())
                selectedImagePath = filePath  // Store the selected image path
                selectedImageView.setImageURI(uri)
            }
        }

        initViewModel()
        initComponents()
        setupObservers()

        return v
    }

    // onViewCreated method to handle UI interactions, calls after the view is created
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val calendarView: CalendarView = view.findViewById(R.id.calendarView)

        // set the ViewModel with selected date from calendar
        viewModel.selectedDate.value = formatDate(Calendar.getInstance())

        // listener updates selectedDate whenever the date selected on the calendar changes
        calendarView.setOnDateChangeListener { _, year, month, dayOfMonth ->
            val calendar = Calendar.getInstance()
            calendar.set(year, month, dayOfMonth)
            viewModel.selectedDate.value = formatDate(calendar)
        }
    }

    //format for display
    private fun formatDate(calendar: Calendar): String {
        val sdf = SimpleDateFormat("dd-MM-yyyy", Locale.UK)
        return sdf.format(calendar.time)
    }

    // function to copy a image file to internal storage - this allows the selected photo to be copied
    // to the apps internal storage - this allows us to keep a copy of the image private to the app
    @Throws(IOException::class)
    fun copyFileToInternalStorage(uri: Uri, newDirName: String, context: Context): String {
        // get filename from uri display_name column
        val returnCursor = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
        val nameIndex = returnCursor!!.getColumnIndexOrThrow(android.provider.OpenableColumns.DISPLAY_NAME)
        returnCursor.moveToFirst()

        // make file to copy into - make directory if required
        val fileName = returnCursor.getString(nameIndex)
        val output = File(context.filesDir, newDirName)
        if (!output.exists()) {
            output.mkdirs()
        }
        val destination = File(output, fileName)

        // input stream reads file and copies the chunks until complete
        context.contentResolver.openInputStream(uri).use { `in` ->
            FileOutputStream(destination).use { out ->
                val buffer = ByteArray(1024)
                var length: Int
                while (`in`!!.read(buffer).also { length = it } > 0) {
                    out.write(buffer, 0, length)
                }
            }
        }

        return destination.absolutePath  // return path of the copied file
    }

    // handle permission request results
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults) // deprecated but still works
        permissionToRecordAccepted = if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        } else {
            false
        }
        if (!permissionToRecordAccepted) return
    }

    // function to setup MediaRecorder with chosen file name
    private fun setupMediaRecorder(fileName: String) {
        val audioFileName = "$fileName.3gp"
        val audioFile = File(requireContext().filesDir, audioFileName)

        recorder = MediaRecorder()
        recorder.apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
            setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
            setOutputFile(audioFile.absolutePath)
            prepare()
        }
    }

    // method to initialize ViewModel
    private fun initViewModel() {
        viewModel = activity?.run {
            ViewModelProvider(this)[NotesViewModel::class.java]
        } ?: throw Exception("Invalid Activity")
    }

    // function to initialise UI components
    private fun initComponents() {
        val valueView = v.findViewById<TextView>(R.id.textView)
        val textNoteLayout: LinearLayout = v.findViewById(R.id.textNoteLayout)
        val audioRecorderLayout: LinearLayout = v.findViewById(R.id.audioRecorderLayout)
        val imageUploaderLayout: LinearLayout = v.findViewById(R.id.imageUploaderLayout)

        initTextNoteComponents(valueView, textNoteLayout)
        initAudioNoteComponents(valueView, audioRecorderLayout)
        initImageNoteComponents(valueView, imageUploaderLayout)

        //  buttons for switching note type
        val textNoteButton = v.findViewById<ImageButton>(R.id.textNoteButton)
        val voiceNoteButton = v.findViewById<ImageButton>(R.id.voiceNoteButton)
        val imageNoteButton = v.findViewById<ImageButton>(R.id.imageNoteButton)

        // listeners for note buttons - switches layout visibility to provide selected function
        textNoteButton.setOnClickListener {
            textNoteLayout.visibility = View.VISIBLE
            audioRecorderLayout.visibility = View.GONE
            imageUploaderLayout.visibility = View.GONE
        }

        voiceNoteButton.setOnClickListener {
            ActivityCompat.requestPermissions(requireActivity(), permissions, REQUEST_RECORD_AUDIO_PERMISSION)
            textNoteLayout.visibility = View.GONE
            audioRecorderLayout.visibility = View.VISIBLE
            imageUploaderLayout.visibility = View.GONE
        }

        imageNoteButton.setOnClickListener {
            textNoteLayout.visibility = View.GONE
            audioRecorderLayout.visibility = View.GONE
            imageUploaderLayout.visibility = View.VISIBLE
        }

        textNoteLayout.visibility = View.VISIBLE
    }

    // initialize components for text notes
    private fun initTextNoteComponents(valueView: TextView, layout: LinearLayout) {
        val noteEditText = v.findViewById<EditText>(R.id.noteEditText)
        val saveButton = v.findViewById<Button>(R.id.saveButton)
        val clearButton = v.findViewById<Button>(R.id.clearButton)

        // listener for text note saving
        saveButton.setOnClickListener {
            val noteText = noteEditText.text.toString()
            val noteType = "text"
            val noteDate = formatForDatabase(valueView.text.toString())

            // add note to db via ViewModel
            viewModel.databaseAdapter.dbHelper.addNote(
                noteText,
                null,
                null,
                noteType,
                noteDate
            )

            Toast.makeText(requireContext(), "Text Note Saved", Toast.LENGTH_SHORT).show() // inform user
            noteEditText.setText("")  // clear text field
            viewModel.updateNoteListFromDatabase()  // update note list
        }

        clearButton.setOnClickListener {
            noteEditText.setText("")
        }
    }

    // initialise components for audio notes
    private fun initAudioNoteComponents(valueView: TextView, layout: LinearLayout) {
        recordButton = v.findViewById<Button>(R.id.recordButton)
        stopRecordButton = v.findViewById<Button>(R.id.stopRecordButton)
        cancelRecordButton = v.findViewById<Button>(R.id.cancelRecordButton)
        fileNameEditText = v.findViewById<EditText>(R.id.fileNameEditText)

        // listeners for audio recording
        stopRecordButton.setOnClickListener {
            Log.d("RECORDING", "stopRecording(valueView, fileNameEditText) Called")
            stopRecording(valueView, fileNameEditText)
        }

        recordButton.setOnClickListener {
            val fileName = requireContext().filesDir.path + "/" + fileNameEditText.text.toString() + ".3gp"

            //if the filename exists, inform the user, else, record
            if (viewModel.databaseAdapter.dbHelper.checkAudioFileNameExists(fileName)) {
                Toast.makeText(requireContext(), "File name already exists!", Toast.LENGTH_SHORT).show()
            } else {
                Log.d("RECORDING", "startRecording(fileNameEditText) Called")
                startRecording(fileNameEditText)
            }
        }

        cancelRecordButton.setOnClickListener {
            Log.d("RECORDING", "cancelRecording() Called")
            cancelRecording()
        }
    }

    // initialise components for image notes
    private fun initImageNoteComponents(valueView: TextView, layout: LinearLayout){
        selectButton = v.findViewById<Button>(R.id.selectButton)
        uploadButton = v.findViewById<Button>(R.id.uploadButton)
        imageTitleEditText = v.findViewById<EditText>(R.id.imageTitleEditText)
        selectedImageView = v.findViewById<ImageView>(R.id.selectedImageView)

        // listeners for image selection / upload
        selectButton.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }

        uploadButton.setOnClickListener {
            if (selectedImagePath != null) {
                saveImageNote(valueView)
                selectedImageView.setImageURI(null)
            } else {
                Toast.makeText(requireContext(), "No image selected", Toast.LENGTH_SHORT).show()
            }
        }
    }


    // function to save image note
    private fun saveImageNote(valueView: TextView) {
        val date = formatForDatabase(valueView.text.toString())
        val note = Note(1, imageTitleEditText.text.toString(), null, selectedImagePath, "image", date)

        // save note to the db using ViewModel
        viewModel.databaseAdapter.dbHelper.addNote(
            note.text_note,
            note.voice_note_path,
            note.image_path,
            note.note_type,
            note.date_created
        )

        // inform user, clear text & update ListView from db
        Toast.makeText(requireContext(), "Image Note Saved", Toast.LENGTH_LONG).show()
        imageTitleEditText.setText("")
        viewModel.updateNoteListFromDatabase()
    }

    // setup observers for ViewModel - text field will be updated when selectedDate changes
    private fun setupObservers() {
        val valueObserver = Observer<String> { newValue ->
            val valueView = v.findViewById<TextView>(R.id.textView)
            valueView.text = newValue
        }
        viewModel.selectedDate.observe(viewLifecycleOwner, valueObserver)
    }

    // function to start audio recording using MediaRecorder
    private fun startRecording(fileNameEditText: EditText) {
        val fileName = fileNameEditText.text.toString()
        setupMediaRecorder(fileName)
        recorder.start()
        Log.d("RECORDING", "recorder started")

        // update UI to reflect state
        recordButton.visibility = View.GONE
        stopRecordButton.visibility = View.VISIBLE
        cancelRecordButton.visibility = View.VISIBLE
    }

    // function to stop audio recording
    private fun stopRecording(valueView: TextView, fileNameEditText: EditText) {
        recorder.stop()
        recordButton.visibility = View.VISIBLE
        stopRecordButton.visibility = View.GONE
        cancelRecordButton.visibility = View.GONE

        val path = requireContext().filesDir.path + "/" + fileNameEditText.text.toString() + ".3gp"
        val date = formatForDatabase(valueView.text.toString())

        val note = Note(1, null, path, null, "audio", date)

        viewModel.databaseAdapter.dbHelper.addNote(
            note.text_note,
            note.voice_note_path,
            note.image_path,
            note.note_type,
            note.date_created
        )
        Log.d("RECORDING", "recorder stopped")

        fileNameEditText.setText("")
        Toast.makeText(requireContext(), "Audio Note Saved", Toast.LENGTH_SHORT).show()
        viewModel.updateNoteListFromDatabase()
    }

    // function to stop audio recording
    private fun cancelRecording() {
        recorder.stop()
        recordButton.visibility = View.VISIBLE
        stopRecordButton.visibility = View.GONE
        cancelRecordButton.visibility = View.GONE

        Log.d("RECORDING", "recording cancelled")
        fileNameEditText.setText("")
        Toast.makeText(requireContext(), "Recording Cancelled", Toast.LENGTH_LONG).show()

        viewModel.updateNoteListFromDatabase()
    }

    private fun formatForDatabase(dateStr: String): String {
        val sdfInput = SimpleDateFormat("dd-MM-yyyy", Locale.UK)
        val sdfOutput = SimpleDateFormat("yyyy-MM-dd", Locale.UK)
        return sdfOutput.format(sdfInput.parse(dateStr))
    }

}
