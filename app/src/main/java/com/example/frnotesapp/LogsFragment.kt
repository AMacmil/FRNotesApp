package com.example.frnotesapp

import android.content.Context
import android.icu.text.SimpleDateFormat
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.ListView
import android.widget.MediaController
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import java.io.File
import java.io.IOException
import java.util.Locale

class LogsFragment : Fragment() {
    private lateinit var viewModel: NotesViewModel
    private lateinit var contentContainer: FrameLayout
    private lateinit var listViewNotes: ListView
    private lateinit var mediaController: MediaController
    private lateinit var player: MediaPlayer

    private var isPlayerInitialized = false  // flag to track initialisation of media player
    private var selPosition: Int = -1  // stores the selected position in the list

    // function to stop audio playback
    private fun stopPlaying() {
        // stop and reset player only if it was initialized and is currently playing
        if (isPlayerInitialized && player.isPlaying) {
            player.stop()
        }
        player.reset()  // reset the player to its idle state
        mediaController.hide()
        isPlayerInitialized = false  // update initialization flag
        Log.d("PLAYBACK", "media player stopped and reset")
    }

    // override onPause method so I can hide the media player / stop playback when app paused
    override fun onPause() {
        super.onPause()
        stopPlaying()
        mediaController.hide()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        viewModel = activity?.run {
            ViewModelProvider(this)[NotesViewModel::class.java]
        } ?: throw Exception("Invalid Activity")

        val v = inflater.inflate(R.layout.logs_fragment, container, false)

        // initialize UI elements
        listViewNotes = v.findViewById(R.id.listViewNotes)
        contentContainer = v.findViewById(R.id.contentContainer)

        player = MediaPlayer()  // initialize media player

        // initialise media controller - timeout is 0 to prevent it auto-hiding
        mediaController = object : MediaController(activity) {
            override fun show(timeout: Int) {
                super.show(0)
            }
        }

        // link controller and player
        mediaController.setMediaPlayer(MediaControllerInterface(player))

        // observer for note list
        val noteListObserver = Observer<List<Note>> { newNoteList ->
            // set up the adapter for the ListView with the note list
            val adapter = NoteAdapter(requireContext(), newNoteList)
            listViewNotes.adapter = adapter
        }

        // observe the ViewModel's note list
        viewModel.noteList.observe(viewLifecycleOwner, noteListObserver)

        // set up click listener for list items
        listViewNotes.setOnItemClickListener { _, _, position, _ ->
            // update content container with the selected note
            val selectedNote = viewModel.noteList.value?.get(position)
            selPosition = position
            // I expect a Note to be selected, and call updateContentContainer with it
            selectedNote?.let { updateContentContainer(it) }
        }

        return v
    }

    // function to update what's displayed in contentContainer - this is generally called when
    // a new note is selected from the list
    private fun updateContentContainer(note: Note) {
        contentContainer.removeAllViews()  // clear existing views

        stopPlaying()  // stop audio (if audio is playing)

        // display different components inside contentContainer depending on selected note type
        when (note.note_type) {
            "text" -> {
                stopPlaying()
                val textView = TextView(requireContext())
                textView.text = "Date: " + formatForDisplay(note.date_created) + "\nContents: " + note.text_note
                val textLayoutParams = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT
                )
                textView.layoutParams = textLayoutParams
                textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
                contentContainer.addView(textView)
            }

            "audio" -> {
                stopPlaying()
                prepareMediaPlayer(note.voice_note_path)
                val filenameTextView = TextView(requireContext()).apply {
                    layoutParams = FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                    )
                    text = "Selected Audio File: " + note.voice_note_path?.split("/")?.last() ?: "No filename"
                }

                filenameTextView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
                contentContainer.addView(filenameTextView)

                // need to create a View as an anchorView for the media player
                val anchorView = View(requireContext()).apply {
                    layoutParams = FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.WRAP_CONTENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                    )
                }

                contentContainer.addView(anchorView)

                // sets the anchorView for the mediaController and shows it without autohide
                mediaController.setAnchorView(anchorView)
                mediaController.show(0)
            }

            "image" -> {
                stopPlaying()
                val imageView = ImageView(requireContext())

                val layoutParams = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    gravity = Gravity.CENTER
                }

                imageView.layoutParams = layoutParams

                // convert uri string to object and set to imageView
                val imageUri = Uri.parse(note.image_path)
                imageView.setImageURI(imageUri)
                contentContainer.addView(imageView)
            }
        }
    }

    // function to prepare MediaPlayer with the provided filepath
    private fun prepareMediaPlayer(voiceNotePath: String?) {
        stopPlaying()  // stop / reset media player before preparing

        try {
            player.apply {
                setDataSource(voiceNotePath) // sets data source for media player

                // listeners allow for automatic behaviour and state management
                setOnPreparedListener {
                    isPlayerInitialized = true
                }
                setOnCompletionListener {
                    isPlayerInitialized = false
                    mediaController.show(0) // reshow to allow replays of media
                }
                prepareAsync()  // prepare player asynchronously
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    // takes date in db format and converts to display format
    private fun formatForDisplay(dateStr: String): String {
        val sdfInput = SimpleDateFormat("yyyy-MM-dd", Locale.UK)
        val sdfOutput = SimpleDateFormat("dd-MM-yyyy", Locale.UK)
        return sdfOutput.format(sdfInput.parse(dateStr))
    }

    // custom Adapter class for displaying notes in a ListView - implements required methods
    inner class NoteAdapter(private val context: Context, private val noteList: List<Note>) :
        BaseAdapter() {

        private val inflater: LayoutInflater =
            context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater

        override fun getCount(): Int = noteList.size

        override fun getItem(position: Int): Any =
            noteList[position]

        override fun getItemId(position: Int): Long =
            position.toLong()

        // function to get a view for each item in the data source using the list_item_note layout
        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = convertView ?: inflater.inflate(
                R.layout.list_item_note,
                parent,
                false
            )  // inflate the custom layout

            // find and set views for date, type, and path
            val dateTextView = view.findViewById<TextView>(R.id.dateTextView)
            val typeTextView = view.findViewById<TextView>(R.id.typeTextView)
            val pathTextView = view.findViewById<TextView>(R.id.pathTextView)

            val note = getItem(position) as Note  // get the note object for the current position

            // set the date / note type text
            dateTextView.text = formatForDisplay(note.date_created)
            typeTextView.text = note.note_type

            // update the pathTextView based on the type of note
            when (note.note_type) {
                "audio" -> pathTextView.text = note.voice_note_path?.substringAfterLast("/")
                "image" -> pathTextView.text = note.text_note
                "text" -> pathTextView.text = note.text_note?.take(15) + "..."
            }

            return view  // return the custom view
        }
    }

    // MediaController interface implementation required for controlling media playback
    internal class MediaControllerInterface(private val mediaPlayer: MediaPlayer?) :
        MediaController.MediaPlayerControl {
        override fun start() = mediaPlayer?.start() ?: Unit
        override fun pause() = mediaPlayer?.pause() ?: Unit
        override fun getDuration() = mediaPlayer?.duration ?: 0
        override fun getCurrentPosition() = mediaPlayer?.currentPosition ?: 0
        override fun seekTo(pos: Int) = mediaPlayer?.seekTo(pos) ?: Unit
        override fun isPlaying() = mediaPlayer?.isPlaying ?: false
        override fun getBufferPercentage() = 0
        override fun canPause() = true
        override fun canSeekBackward() = true
        override fun canSeekForward() = true
        override fun getAudioSessionId() =
            0  // return 0 if there's no audio session, or the session ID if available
    }
}