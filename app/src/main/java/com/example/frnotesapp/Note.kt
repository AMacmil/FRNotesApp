package com.example.frnotesapp

// data class representing a note, made of a unique id, nullable strings for note contents / file
// paths, strings for note type and date
data class Note(
    val id: Int,
    val text_note: String?,
    val voice_note_path: String?,
    val image_path: String?,
    val note_type: String,
    val date_created: String
)
