package com.example.frnotesapp

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.SQLException
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log

class DBAdapter(private val dbContext: Context) {
    // internal DatabaseHelper for managing db creation
    internal val dbHelper: DatabaseHelper = DatabaseHelper(dbContext)

    // SQLiteDatabase instance for database operations - initialised later
    private lateinit var db: SQLiteDatabase

    // companion object to hold database schema info
    companion object {
        private const val DATABASE_VERSION = 1
        private const val DATABASE_NAME = "NotesDB"
        const val TABLE_NAME = "notes"
        const val COLUMN_ID = "id"
        const val COLUMN_TEXT_NOTE = "text_note"
        const val COLUMN_VOICE_NOTE_PATH = "voice_note_path"
        const val COLUMN_IMAGE_PATH = "image_path"
        const val COLUMN_NOTE_TYPE = "note_type"
        const val COLUMN_DATE_CREATED = "date_created"
    }

    // function to open the database
    @Throws(SQLException::class)
    fun open(): DBAdapter {
        db = dbHelper.writableDatabase
        return this
    }

    // function to drop the notes table and all data
    fun dropTable() {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_NAME")
    }

    // function to recreate the notes table.
    fun recreateTable() {
        // drop old table if it exists
        db.execSQL("DROP TABLE IF EXISTS $TABLE_NAME")
        val CREATE_TABLE_QUERY =
            "CREATE TABLE $TABLE_NAME (" +
                    "$COLUMN_ID INTEGER PRIMARY KEY AUTOINCREMENT, " +
                    "$COLUMN_TEXT_NOTE TEXT, " +
                    "$COLUMN_VOICE_NOTE_PATH TEXT, " +
                    "$COLUMN_IMAGE_PATH TEXT, " +
                    "$COLUMN_NOTE_TYPE TEXT, " +
                    "$COLUMN_DATE_CREATED DATETIME DEFAULT CURRENT_TIMESTAMP)"
        // execute create table command
        db.execSQL(CREATE_TABLE_QUERY)
    }

    // inner class DatabaseHelper for database creation and versioning
    internal inner class DatabaseHelper(context: Context?) :
        SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {
        override fun onCreate(db: SQLiteDatabase) {
            val CREATE_TABLE_QUERY =
                "CREATE TABLE $TABLE_NAME (" +
                        "$COLUMN_ID INTEGER PRIMARY KEY AUTOINCREMENT, " +
                        "$COLUMN_TEXT_NOTE TEXT, " +
                        "$COLUMN_VOICE_NOTE_PATH TEXT, " +
                        "$COLUMN_IMAGE_PATH TEXT, " +
                        "$COLUMN_NOTE_TYPE TEXT, " +
                        "$COLUMN_DATE_CREATED DATETIME DEFAULT CURRENT_TIMESTAMP)"
            db.execSQL(CREATE_TABLE_QUERY)
        }

        // called when the database needs to be upgraded
        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            db.execSQL("DROP TABLE IF EXISTS $TABLE_NAME")
            onCreate(db)
        }

        // called when the database has been opened - used for logging
        override fun onOpen(db: SQLiteDatabase?) {
            super.onOpen(db)
            Log.d("DatabaseHelper", "Database has been opened")
        }

        // function to add a new note to the database
        fun addNote(
            textNote: String?,
            voiceNotePath: String?,
            imagePath: String?,
            noteType: String,
            noteDate: String
        ) {
            val values = ContentValues() // creating ContentValues object for inserting data.
            values.put(COLUMN_TEXT_NOTE, textNote)
            values.put(COLUMN_VOICE_NOTE_PATH, voiceNotePath)
            values.put(COLUMN_IMAGE_PATH, imagePath)
            values.put(COLUMN_NOTE_TYPE, noteType)
            values.put(COLUMN_DATE_CREATED, noteDate)

            // get writable db instance & insert values into table
            val db = this.writableDatabase
            db.insert(TABLE_NAME, null, values)
            db.close()
        }

        // retrieve all notes from the db
        fun getAllNotes(): Cursor {
            val db = this.readableDatabase
            // SQL query to select all notes ordered by date
            return db.rawQuery("SELECT * FROM $TABLE_NAME ORDER BY $COLUMN_DATE_CREATED DESC", null)
        }

        // method to prevent duplicate file-paths
        fun checkAudioFileNameExists(fileName: String): Boolean {
            val db = this.readableDatabase
            val query = "SELECT * FROM $TABLE_NAME WHERE $COLUMN_VOICE_NOTE_PATH = ?"
            val cursor = db.rawQuery(query, arrayOf(fileName))
            val exists = cursor.count > 0
            cursor.close()
            return exists
        }
    }
}