package com.example.frnotesapp

import android.content.Context
import android.database.Cursor
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class NotesViewModel : ViewModel() {
    // DatabaseAdapter to handle db operations
    lateinit var databaseAdapter: DBAdapter

    // MutableLiveData to store selected date - mutable data allows for dynamically changing values
    private val _selectedDate = MutableLiveData<String>()
    // public data for selectedDate allowing it's access from elsewhere
    val selectedDate: MutableLiveData<String>
        get() = _selectedDate

    // MutableLiveData to store the list of notes
    private val _noteList = MutableLiveData<List<Note>>()
    // public data for noteList, same as above
    val noteList: MutableLiveData<List<Note>>
        get() = _noteList

    // initialize DatabaseAdapter & open
    fun initDatabase(context: Context) {
        databaseAdapter = DBAdapter(context)
        databaseAdapter.open()
        // uncomment to reset db
        /*databaseAdapter.dropTable()
        databaseAdapter.recreateTable()*/
    }

    // function to update note list from db
    fun updateNoteListFromDatabase() {
        val dbNotes = mutableListOf<Note>() // empty list to store notes

        // Cursor to iterate over the notes
        val cursor: Cursor = databaseAdapter.dbHelper.getAllNotes()

        // if query returned any rows
        if (cursor.moveToFirst()) {
            do {
                // retrieve column indexes
                val idIndex = cursor.getColumnIndex(DBAdapter.COLUMN_ID)
                val textNoteIndex = cursor.getColumnIndex(DBAdapter.COLUMN_TEXT_NOTE)
                val voiceNotePathIndex = cursor.getColumnIndex(DBAdapter.COLUMN_VOICE_NOTE_PATH)
                val imagePathIndex = cursor.getColumnIndex(DBAdapter.COLUMN_IMAGE_PATH)
                val noteTypeIndex = cursor.getColumnIndex(DBAdapter.COLUMN_NOTE_TYPE)
                val dateCreatedIndex = cursor.getColumnIndex(DBAdapter.COLUMN_DATE_CREATED)

                // check all column indices are valid / not -1
                if (idIndex != -1 && textNoteIndex != -1 && voiceNotePathIndex != -1 &&
                    imagePathIndex != -1 && noteTypeIndex != -1 && dateCreatedIndex != -1) {
                    // retrieve data from cursor
                    val id = cursor.getInt(idIndex)
                    val textNote = cursor.getString(textNoteIndex)
                    val voiceNotePath = cursor.getString(voiceNotePathIndex)
                    val imagePath = cursor.getString(imagePathIndex)
                    val noteType = cursor.getString(noteTypeIndex)
                    val dateCreated = cursor.getString(dateCreatedIndex)

                    // add note to the list
                    dbNotes.add(Note(id, textNote, voiceNotePath, imagePath, noteType, dateCreated))
                }
            } while (cursor.moveToNext()) // move to next row in cursor if possible
        }
        cursor.close()
        // update _noteList with the list of notes
        _noteList.value = dbNotes
    }

    // init block to set default value for _selectedDate
    init {
        val sdf = SimpleDateFormat("dd-MM-yyyy", Locale.UK)
        // set current date as default value for _selectedDate
        _selectedDate.value = sdf.format(Date())
    }
}