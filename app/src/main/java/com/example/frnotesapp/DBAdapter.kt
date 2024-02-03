package com.example.frnotesapp

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.SQLException
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log
import java.nio.ByteBuffer

class DBAdapter(private val dbContext: Context) {
    // internal DatabaseHelper for managing db creation
    internal val dbHelper: DatabaseHelper = DatabaseHelper(dbContext)

    // SQLiteDatabase instance for database operations - initialised later
    private lateinit var db: SQLiteDatabase

    // companion object to hold database schema info
    companion object {
        private const val DATABASE_VERSION = 2
        private const val DATABASE_NAME = "AppDB"

        const val TABLE_NAME = "notes"
        const val COLUMN_ID = "id"
        const val COLUMN_TEXT_NOTE = "text_note"
        const val COLUMN_VOICE_NOTE_PATH = "voice_note_path"
        const val COLUMN_IMAGE_PATH = "image_path"
        const val COLUMN_NOTE_TYPE = "note_type"
        const val COLUMN_DATE_CREATED = "date_created"

        // table and column names for feature vectors
        const val TABLE_FEATURES = "features"
        const val COLUMN_FEATURE_ID = "id"
        const val COLUMN_FEATURE_VECTOR = "feature_vector"
    }

    // function to open the database
    @Throws(SQLException::class)
    fun open(): DBAdapter {
        db = dbHelper.writableDatabase
        return this
    }

    fun close() {
        dbHelper.close()
    }

    // function to drop the notes table and all data
    fun dropTable() {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_NAME")

        db.execSQL("DROP TABLE IF EXISTS $TABLE_FEATURES")
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
            createNotesTable(db)
            createFeaturesTable(db)
        }

        private fun createNotesTable(db: SQLiteDatabase) {
            val CREATE_TABLE_QUERY =
                """
                CREATE TABLE $TABLE_NAME (
                    $COLUMN_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    $COLUMN_TEXT_NOTE TEXT,
                    $COLUMN_VOICE_NOTE_PATH TEXT,
                    $COLUMN_IMAGE_PATH TEXT,
                    $COLUMN_NOTE_TYPE TEXT,
                    $COLUMN_DATE_CREATED DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """.trimIndent()
            db.execSQL(CREATE_TABLE_QUERY)
        }

        private fun createFeaturesTable(db: SQLiteDatabase) {
            val CREATE_FEATURES_TABLE_QUERY =
                """
                CREATE TABLE $TABLE_FEATURES (
                    $COLUMN_FEATURE_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    $COLUMN_FEATURE_VECTOR BLOB
                )
                """.trimIndent()
            db.execSQL(CREATE_FEATURES_TABLE_QUERY)
        }

        // called when the database needs to be upgraded
        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            db.execSQL("DROP TABLE IF EXISTS $TABLE_NAME")
            onCreate(db)

            // Drop the features table if exists
            db.execSQL("DROP TABLE IF EXISTS $TABLE_FEATURES")
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

        fun addFeatureVector(featureVector: FloatArray) {
            val vectorBlob = floatArrayToBlob(featureVector)
            val values = ContentValues()
            values.put(COLUMN_FEATURE_VECTOR, vectorBlob)

            val db = this.writableDatabase
            db.insert(TABLE_FEATURES, null, values)
        }

        fun getFeatureVector(id: Int): FloatArray? {
            val db = this.readableDatabase
            val cursor = db.rawQuery("SELECT $COLUMN_FEATURE_VECTOR FROM $TABLE_FEATURES WHERE $COLUMN_FEATURE_ID = ?", arrayOf(id.toString()))

            if (cursor.moveToFirst()) {
                val blob = cursor.getBlob(cursor.getColumnIndexOrThrow(COLUMN_FEATURE_VECTOR))
                cursor.close()
                return blobToFloatArray(blob)
            }
            cursor.close()
            return null
        }

        private fun floatArrayToBlob(array: FloatArray): ByteArray {
            val buffer = ByteBuffer.allocate(array.size * 4)
            buffer.asFloatBuffer().put(array)
            return buffer.array()
        }

        private fun blobToFloatArray(blob: ByteArray): FloatArray {
            val buffer = ByteBuffer.wrap(blob)
            val floatArray = FloatArray(blob.size / 4)
            buffer.asFloatBuffer().get(floatArray)
            return floatArray
        }
    }
    }
